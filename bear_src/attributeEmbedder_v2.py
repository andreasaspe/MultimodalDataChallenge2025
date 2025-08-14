import torch
import torch.nn as nn

class AttributeEmbedderV2(nn.Module):
    """
    Same interface as your AttributeEmbedder, but:
      - month/day -> sin/cos encodings
      - geo -> Fourier features
      - returns the SAME (B, 7*E) shape so your fusion model still works.
    """
    def __init__(self, num_habitats, num_substrates, num_camera_models, num_camera_makers,
                 num_embedding_dims=64, fourier_features=8, geo_scale=10.0):
        super().__init__()
        E = num_embedding_dims
        self.habitat_embedding = nn.Embedding(num_habitats, E)
        self.substrate_embedding = nn.Embedding(num_substrates, E)
        self.camera_model_embedding = nn.Embedding(num_camera_models, E)
        self.camera_maker_embedding = nn.Embedding(num_camera_makers, E)
        self.max_Latitude = 57.739133
        self.min_Latitude = 54.56094
        self.max_Longitude = 15.14406
        self.min_Longitude = 8.08042

        # project sin/cos pairs to E
        self.time_proj = nn.Linear(4, E)   # [sin(mon), cos(mon), sin(day), cos(day)]
        # random Fourier features for lat/lon (normalized to [-1,1] before this)
        #self.B = nn.Parameter(torch.randn(2, fourier_features) * geo_scale, requires_grad=False)
        self.geo_proj = nn.Linear(2, E)  # [lat, lon] -> [sin(lat), cos(lat), sin(lon), cos(lon)] * F
        #self.geo_proj = nn.Linear(4 * fourier_features, E)  # sin/cos for each feature

        self.dropout = nn.Dropout(p=0.0)  # mild regularization on attributes

    def _time_embed(self, month, day):
        # month in [0..12], day in [0..24] with a "missing" bucket at the end
        mon = month.clamp(min=0, max=11).float()  # treat missing as 11 -> becomes arbitrary; ok since we drop out attrs
        day  = day.clamp(min=1, max=31).float()
        mon_sin = torch.sin(2 * torch.pi * (mon / 12.0))
        mon_cos = torch.cos(2 * torch.pi * (mon / 12.0))
        day_sin  = torch.sin(2 * torch.pi * (day / 31.0))
        day_cos  = torch.cos(2 * torch.pi * (day / 31.0))
        x = torch.stack([mon_sin, mon_cos, day_sin, day_cos], dim=-1)  # (B,4)
        return self.time_proj(x)

    def _geo_embed(self, latitude, longitude):
        # normalize to roughly [-1,1]; handle missing as zeros
        #lat = torch.clamp(latitude / 90.0, -1.0, 1.0).unsqueeze(-1)    # (B,1)
        #lon = torch.clamp(longitude / 180.0, -1.0, 1.0).unsqueeze(-1)  # (B,1)
        lat = (latitude - self.min_Latitude) / (self.max_Latitude - self.min_Latitude) * 2 - 1
        lon = (longitude - self.min_Longitude) / (self.max_Longitude - self.min_Longitude) * 2 - 1
        lat = lat.clamp(-1.0, 1.0).unsqueeze(-1)  # (B,1)
        lon = lon.clamp(-1.0, 1.0).unsqueeze(-1)  # (B,1)
        geo_feats = torch.cat([lat, lon], dim=-1)  # (B,2)
        #X = torch.cat([lat, lon], dim=-1)                               # (B,2)
        # random Fourier features
        #proj = X @ self.B  # (B, 2) x (2, F) -> (B, F)
        #geo_feats = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # (B, 2F)
        # duplicate for each of lat/lon pairs -> 4F
        #geo_feats = torch.cat([geo_feats, geo_feats], dim=-1)
        #return self.geo_proj(geo_feats)
        return self.geo_proj(geo_feats)  # (B,E)

    def forward(self, habitat, substrate, month, day, camera_model, camera_maker, latitude, longitude):
        h = self.habitat_embedding(habitat)
        s = self.substrate_embedding(substrate)
        t = self._time_embed(month, day)
        cmod = self.camera_model_embedding(camera_model)
        cmak = self.camera_maker_embedding(camera_maker)
        g = self._geo_embed(latitude, longitude)

        # we still produce 7 tokens * E so the fusion model remains unchanged
        tokens = torch.stack([h, s, t, cmod, cmak, g], dim=1)  # (B,7,E)
        tokens = self.dropout(tokens)
        return tokens.reshape(tokens.size(0), -1)  # (B, 7*E)