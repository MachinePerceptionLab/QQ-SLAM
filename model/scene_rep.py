# package imports
import torch
import torch.nn as nn

# Local imports
from .encodings import get_encoder
from .decoder import ColorSDFNet, ColorSDFNet_v2
from .utils import sample_pdf, batchify, get_sdf_loss, mse2psnr, compute_loss, grid_sample_3d
import distributed as dist_fn
from torch.nn import functional as F

class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        # embed = torch.randn(dim, n_embed)
        embed = torch.rand(dim, n_embed) * (2.0 / self.n_embed) - 1.0 / self.n_embed
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind0 = (-dist).max(1)
        # embed_onehot = F.one_hot(embed_ind0, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind0.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_ind0 = embed_ind0.to(torch.int64)
            # Calculate embed_onehot_sum
            embed_onehot_sum = torch.zeros(self.n_embed, dtype=flatten.dtype, device=embed_ind.device)
            embed_onehot_sum.scatter_add_(0, embed_ind0, torch.ones_like(embed_ind0, dtype=flatten.dtype))

            # Calculate embed_sum 
            embed_sum = torch.zeros((self.n_embed, flatten.size(1)), dtype=flatten.dtype, device=embed_ind.device)
            embed_sum.index_add_(0, embed_ind0, flatten[:embed_ind0.size(0)])
            embed_sum = embed_sum.T

            # embed_onehot_sum1 = embed_onehot.sum(0)
            # embed_sum1 = flatten.transpose(0, 1) @ embed_onehot

            dist_fn.all_reduce(embed_onehot_sum)
            dist_fn.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, torch.zeros((1)).cuda(), embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class JointEncoding(nn.Module):
    def __init__(self, config, bound_box):
        super(JointEncoding, self).__init__()
        self.config = config
        self.bounding_box = bound_box
        self.get_resolution()
        self.get_encoding(config)
        self.get_decoder(config)

        self.quantize_b = Quantize(self.input_ch, self.config["vq_dim"])

    def get_resolution(self):
        '''
        Get the resolution of the grid
        '''
        dim_max = (self.bounding_box[:,1] - self.bounding_box[:,0]).max()
        if self.config['grid']['voxel_sdf'] > 10:
            self.resolution_sdf = self.config['grid']['voxel_sdf']
        else:
            self.resolution_sdf = int(dim_max / self.config['grid']['voxel_sdf'])
        
        if self.config['grid']['voxel_color'] > 10:
            self.resolution_color = self.config['grid']['voxel_color']
        else:
            self.resolution_color = int(dim_max / self.config['grid']['voxel_color'])
        
        print('SDF resolution:', self.resolution_sdf)

    def get_encoding(self, config):
        '''
        Get the encoding of the scene representation
        '''
        # Coordinate encoding
        self.embedpos_fn, self.input_ch_pos = get_encoder(config['pos']['enc'], n_bins=self.config['pos']['n_bins'])

        # Sparse parametric encoding (SDF)
        self.embed_fn, self.input_ch = get_encoder(config['grid']['enc'], log2_hashmap_size=config['grid']['hash_size'], desired_resolution=self.resolution_sdf)

        # Sparse parametric encoding (Color)
        if not self.config['grid']['oneGrid']:
            print('Color resolution:', self.resolution_color)
            self.embed_fn_color, self.input_ch_color = get_encoder(config['grid']['enc'], log2_hashmap_size=config['grid']['hash_size'], desired_resolution=self.resolution_color)

    def get_decoder(self, config):
        '''
        Get the decoder of the scene representation
        '''
        if not self.config['grid']['oneGrid']:
            self.decoder = ColorSDFNet(config, input_ch=self.input_ch, input_ch_pos=self.input_ch_pos)
        else:
            self.decoder = ColorSDFNet_v2(config, input_ch=self.input_ch, input_ch_pos=self.input_ch_pos)
        
        self.color_net = batchify(self.decoder.color_net, None)
        self.sdf_net = batchify(self.decoder.sdf_net, None)

    def sdf2weights(self, sdf, z_vals, args=None):
        '''
        Convert signed distance function to weights.

        Params:
            sdf: [N_rays, N_samples]
            z_vals: [N_rays, N_samples]
        Returns:
            weights: [N_rays, N_samples]
        '''
        weights = torch.sigmoid(sdf / args['training']['trunc']) * torch.sigmoid(-sdf / args['training']['trunc'])

        signs = sdf[:, 1:] * sdf[:, :-1]
        mask = torch.where(signs < 0.0, torch.ones_like(signs), torch.zeros_like(signs))
        inds = torch.argmax(mask, axis=1)
        inds = inds[..., None]
        z_min = torch.gather(z_vals, 1, inds) # The first surface
        mask = torch.where(z_vals < z_min + args['data']['sc_factor'] * args['training']['trunc'], torch.ones_like(z_vals), torch.zeros_like(z_vals))

        weights = weights * mask
        return weights / (torch.sum(weights, axis=-1, keepdims=True) + 1e-8)
    
    def raw2outputs(self, raw, z_vals, white_bkgd=False):
        '''
        Perform volume rendering using weights computed from sdf.

        Params:
            raw: [N_rays, N_samples, 4]
            z_vals: [N_rays, N_samples]
        Returns:
            rgb_map: [N_rays, 3]
            disp_map: [N_rays]
            acc_map: [N_rays]
            weights: [N_rays, N_samples]
        '''
        rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
        weights = self.sdf2weights(raw[..., 3], z_vals, args=self.config)
        rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

        depth_map = torch.sum(weights * z_vals, -1)
        depth_var = torch.sum(weights * torch.square(z_vals - depth_map.unsqueeze(-1)), dim=-1)
        disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
        acc_map = torch.sum(weights, -1)

        if white_bkgd:
            rgb_map = rgb_map + (1.-acc_map[...,None])

        return rgb_map, disp_map, acc_map, weights, depth_map, depth_var
    
    def query_sdf(self, query_points, tsdf_numpy=None, tsdf_bounds=None, return_geo=False, embed=False,return_id = False):

        '''
        Get the SDF value of the query points
        Params:
            query_points: [N_rays, N_samples, 3]
        Returns:
            sdf: [N_rays, N_samples]
            geo_feat: [N_rays, N_samples, channel]
        '''
        inputs_flat = torch.reshape(query_points, [-1, query_points.shape[-1]])
  
        embedded = self.embed_fn(inputs_flat)
        if embed:
            return torch.reshape(embedded, list(query_points.shape[:-1]) + [embedded.shape[-1]])

        embedded_pos = self.embedpos_fn(inputs_flat)
        eval_tsdf = self.sample_point_features_by_linear_interp(query_points.clone(), tsdf_numpy, tsdf_bounds)
        if self.config["use_vq"]: 
            # quant_b = self.quantize_conv_b(embedded[...,None,None])
            quant_b, diff_b, pairwise_loss, embed_ind = self.quantize_b(embedded)
            out = self.sdf_net(eval_tsdf,torch.cat([quant_b, embedded_pos], dim=-1))
        else:
            pairwise_loss = torch.tensor([0.]).to(embedded.device).mean()
            out = self.sdf_net(eval_tsdf, torch.cat([embedded, embedded_pos], dim=-1))
        sdf, geo_feat = out[..., :1], out[..., 1:]

        sdf = torch.reshape(sdf, list(query_points.shape[:-1]))
        if not return_geo:
            if self.config["use_vq"] and return_id:
                return quant_b, embed_ind
            return sdf, pairwise_loss
        geo_feat = torch.reshape(geo_feat, list(query_points.shape[:-1]) + [geo_feat.shape[-1]])

        return sdf, geo_feat
    
    def query_color(self, query_points,tsdf_numpy=None, tsdf_bounds=None):
        return torch.sigmoid(self.query_color_sdf(query_points,tsdf_numpy=tsdf_numpy, tsdf_bounds=tsdf_bounds)[0][..., :3])
      
    def query_color_sdf(self, query_points,tsdf_numpy=None, tsdf_bounds=None,view_dirs=None ):
        '''
        Query the color and sdf at query_points.
        Params:
            query_points: [N_rays, N_samples, 3]
        Returns:
            raw: [N_rays, N_samples, 4]
        '''
        # query_points = self.tsdf_loc(query_points)
        inputs_flat = torch.reshape(query_points, [-1, query_points.shape[-1]])

        embed = self.embed_fn(inputs_flat)
        eval_tsdf = self.sample_point_features_by_linear_interp(query_points.clone(), tsdf_numpy, tsdf_bounds)
        if self.config["use_vq"]: 
            # quant_b = self.quantize_conv_b(embed[...,None,None])
            quant_b, diff_b, pairwise_loss, embed_ind = self.quantize_b(embed)
            embe_pos = self.embedpos_fn(inputs_flat)
            if not self.config['grid']['oneGrid']:
                embed_color = self.embed_fn_color(inputs_flat)
                return self.decoder(eval_tsdf, quant_b, embe_pos, embed_color),diff_b,pairwise_loss
            return self.decoder(eval_tsdf, quant_b, embe_pos,view_dirs),diff_b,pairwise_loss

        else:
            embe_pos = self.embedpos_fn(inputs_flat)
            if not self.config['grid']['oneGrid']:
                embed_color = self.embed_fn_color(inputs_flat)
                return self.decoder(eval_tsdf, embed, embe_pos, embed_color),torch.tensor([0.]).to(embed.device).mean(),torch.tensor([0.]).to(embed.device).mean()
            return self.decoder(eval_tsdf, embed, embe_pos,view_dirs), torch.tensor([0.]).to(embed.device).mean(),torch.tensor([0.]).to(embed.device).mean()
    
    def run_network(self, inputs,tsdf_numpy=None, tsdf_bounds=None,view_dirs=None):
        """
        Run the network on a batch of inputs.

        Params:
            inputs: [N_rays, N_samples, 3]
        Returns:
            outputs: [N_rays, N_samples, 4]
        """
        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
        
        # Normalize the input to [0, 1] (TCNN convention)
        if self.config['grid']['tcnn_encoding']:
            inputs_flat = (inputs_flat - self.bounding_box[:, 0]) / (self.bounding_box[:, 1] - self.bounding_box[:, 0])

        outputs_flat,diff_b,pairwise_loss = batchify(self.query_color_sdf, None)(inputs_flat,tsdf_numpy=tsdf_numpy, tsdf_bounds=tsdf_bounds,view_dirs=view_dirs)
        outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])

        return outputs,diff_b,pairwise_loss
    
    def render_surface_color(self, rays_o, normal):
        '''
        Render the surface color of the points.
        Params:
            points: [N_rays, 1, 3]
            normal: [N_rays, 3]
        '''
        n_rays = rays_o.shape[0]
        trunc = self.config['training']['trunc']
        z_vals = torch.linspace(-trunc, trunc, steps=self.config['training']['n_range_d']).to(rays_o)
        z_vals = z_vals.repeat(n_rays, 1)
        # Run rendering pipeline
        
        pts = rays_o[...,:] + normal[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
        if self.config["view_dirs"]:
            view_dirs = F.normalize(normal, dim=-1)[:, None, :].repeat(1, pts.shape[1], 1)
            raw = self.run_network(pts,view_dirs = view_dirs)[0]
        else:
            raw = self.run_network(pts)[0]
        rgb, disp_map, acc_map, weights, depth_map, depth_var = self.raw2outputs(raw, z_vals, self.config['training']['white_bkgd'])
        return rgb
    
    def render_rays(self, rays_o, rays_d, tsdf_numpy=None, tsdf_bounds=None,target_d=None):
        '''
        Params:
            rays_o: [N_rays, 3]
            rays_d: [N_rays, 3]
            target_d: [N_rays, 1]

        '''
        n_rays = rays_o.shape[0]

        # Sample depth
        if target_d is not None:
            z_samples = torch.linspace(-self.config['training']['range_d'], self.config['training']['range_d'], steps=self.config['training']['n_range_d']).to(target_d) 
            z_samples = z_samples[None, :].repeat(n_rays, 1) + target_d
            z_samples[target_d.squeeze()<=0] = torch.linspace(self.config['cam']['near'], self.config['cam']['far'], steps=self.config['training']['n_range_d']).to(target_d) 

            if self.config['training']['n_samples_d'] > 0:
                z_vals = torch.linspace(self.config['cam']['near'], self.config['cam']['far'], self.config['training']['n_samples_d'])[None, :].repeat(n_rays, 1).to(rays_o)
                z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            else:
                z_vals = z_samples
        else:
            z_vals = torch.linspace(self.config['cam']['near'], self.config['cam']['far'], self.config['training']['n_samples']).to(rays_o)
            z_vals = z_vals[None, :].repeat(n_rays, 1) # [n_rays, n_samples]

        # Perturb sampling depths
        if self.config['training']['perturb'] > 0.:
            mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            upper = torch.cat([mids, z_vals[...,-1:]], -1)
            lower = torch.cat([z_vals[...,:1], mids], -1)
            z_vals = lower + (upper - lower) * torch.rand(z_vals.shape).to(rays_o)

        # Run rendering pipeline
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
        if self.config["view_dirs"]:
            view_dirs = F.normalize(rays_d, dim=-1)[:, None, :].repeat(1, pts.shape[1], 1)
            raw,diff_b,pairwise_loss = self.run_network(pts,tsdf_numpy=tsdf_numpy, tsdf_bounds=tsdf_bounds,view_dirs=view_dirs)
        else:
            raw,diff_b,pairwise_loss = self.run_network(pts,tsdf_numpy=tsdf_numpy, tsdf_bounds=tsdf_bounds)
        rgb_map, disp_map, acc_map, weights, depth_map, depth_var = self.raw2outputs(raw, z_vals, self.config['training']['white_bkgd'])

        # Importance sampling
        if self.config['training']['n_importance'] > 0:

            rgb_map_0, disp_map_0, acc_map_0, depth_map_0, depth_var_0 = rgb_map, disp_map, acc_map, depth_map, depth_var

            z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], self.config['training']['n_importance'], det=(self.config['training']['perturb']==0.))
            z_samples = z_samples.detach()

            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]
            if self.config["view_dirs"]:
                view_dirs = F.normalize(rays_d, dim=-1)[:, None, :].repeat(1, pts.shape[1], 1)
                raw = self.run_network(pts,tsdf_numpy=tsdf_numpy, tsdf_bounds=tsdf_bounds,view_dirs=view_dirs)
            else:
                raw = self.run_network(pts,tsdf_numpy=tsdf_numpy, tsdf_bounds=tsdf_bounds)
            rgb_map, disp_map, acc_map, weights, depth_map, depth_var = self.raw2outputs(raw, z_vals, self.config['training']['white_bkgd'])

        # Return rendering outputs
        ret = {'rgb' : rgb_map, 'depth' :depth_map, 
               'disp_map' : disp_map, 'acc_map' : acc_map, 
               'depth_var':depth_var,'diff':diff_b,"pairloss":pairwise_loss}
        ret = {**ret, 'z_vals': z_vals}

        ret['raw'] = raw

        if self.config['training']['n_importance'] > 0:
            ret['rgb0'] = rgb_map_0
            ret['disp0'] = disp_map_0
            ret['acc0'] = acc_map_0
            ret['depth0'] = depth_map_0
            ret['depth_var0'] = depth_var_0
            ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)

        return ret
    
    def forward(self, rays_o, rays_d, target_rgb, target_d, tsdf_numpy, tsdf_bounds, global_step=0):
        '''
        Params:
            rays_o: ray origins (Bs, 3)
            rays_d: ray directions (Bs, 3)
            frame_ids: use for pose correction (Bs, 1)
            target_rgb: rgb value (Bs, 3)
            target_d: depth value (Bs, 1)
            c2w_array: poses (N, 4, 4) 
             r r r tx
             r r r ty
             r r r tz
        '''

        # Get render results
        rend_dict = self.render_rays(rays_o, rays_d, tsdf_numpy=tsdf_numpy, tsdf_bounds=tsdf_bounds,target_d=target_d)

        # if not self.training:
        #     return rend_dict
        
        # Get depth and rgb weights for loss
        valid_depth_mask = (target_d.squeeze() > 0.) * (target_d.squeeze() < self.config['cam']['depth_trunc'])
        rgb_weight = valid_depth_mask.clone().unsqueeze(-1)
        rgb_weight[rgb_weight==0] = self.config['training']['rgb_missing']

        # Get render loss
        rgb_loss = compute_loss(rend_dict["rgb"]*rgb_weight, target_rgb*rgb_weight)
        psnr = mse2psnr(rgb_loss)
        depth_loss = compute_loss(rend_dict["depth"].squeeze()[valid_depth_mask], target_d.squeeze()[valid_depth_mask])

        if 'rgb0' in rend_dict:
            rgb_loss += compute_loss(rend_dict["rgb0"]*rgb_weight, target_rgb*rgb_weight)
            depth_loss += compute_loss(rend_dict["depth0"][valid_depth_mask], target_d.squeeze()[valid_depth_mask])
        
        # Get sdf loss
        z_vals = rend_dict['z_vals']  # [N_rand, N_samples + N_importance]
        sdf = rend_dict['raw'][..., -1]  # [N_rand, N_samples + N_importance]
        truncation = self.config['training']['trunc'] * self.config['data']['sc_factor']
        fs_loss, sdf_loss = get_sdf_loss(z_vals, target_d, sdf, truncation, 'l2', grad=None)         
        

        ret = {
            "rgb": rend_dict["rgb"],
            "depth": rend_dict["depth"],
            "rgb_loss": rgb_loss,
            "depth_loss": depth_loss,
            "sdf_loss": sdf_loss,
            "fs_loss": fs_loss,
            "psnr": psnr,
            "diff": rend_dict["diff"],
            "pairloss": rend_dict["pairloss"],
        }

        return ret

    def sample_point_features_by_linear_interp(
        self, coords,tsdf_numpy, origin
    ):
        """
        coords: BN3
        voxel_feats: BFXYZ
        voxel_valid: BXYZ
        grid_origin: B3
        """
        coords = coords.squeeze(-2)[None]
        coords = coords * (self.bounding_box[:, 1] - self.bounding_box[:, 0]) + self.bounding_box[:, 0]
        # # grid_origin = torch.tensor([[ 0.0768, -0.1560, -0.2050]], device=coords.device)
        # # grid_origin = origin[None]
        tsdf_numpy = tsdf_numpy.permute(0,1,4,3,2)
        grid_origin = origin[:,0][None]
        voxel_size = 4./256 #0.02 #0.04
        crop_size_m = (
            torch.tensor(tsdf_numpy.shape[2:], device=coords.device)
            * voxel_size
        )
        coords = (
            coords - grid_origin[:, None] + voxel_size / 2
        ) / crop_size_m * 2 - 1

        # point_feats = torch.nn.functional.grid_sample(
        #     tsdf_numpy, #[None, None]
        #     grid[None, None, :, :, [2, 1, 0]],
        #     align_corners=False,
        #     mode="bilinear",
        #     padding_mode="zeros",
        # )[0,:,0]
        point_feats = grid_sample_3d(tsdf_numpy,coords[None, None, :, :, [2, 1, 0]])[0,:,0]
        return point_feats #(48,n,1) (48,6144,96)