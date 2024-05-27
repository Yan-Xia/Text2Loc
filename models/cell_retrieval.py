from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from models.language_encoder import LanguageEncoder
from models.object_encoder import ObjectEncoder


class CellRetrievalNetwork(torch.nn.Module):
    def __init__(
        self, known_classes: List[str], known_colors: List[str], args
    ):
        """Module for global place recognition.
        Implemented as a text branch (language encoder) and a 3D submap branch (object encoder).
        The 3D submap branch aggregates information about a varying count of multiple objects through Attention.
        """
        super(CellRetrievalNetwork, self).__init__()
        self.embed_dim = args.coarse_embed_dim

        """
        3D submap branch
        """

        # CARE: possibly handle variation in forward()!

        self.object_encoder = ObjectEncoder(args.coarse_embed_dim, known_classes, known_colors, args)
        self.object_size = args.object_size
        num_heads = args.object_inter_module_num_heads
        num_layers = args.object_inter_module_num_layers

        self.obj_inter_module = nn.ModuleList([nn.TransformerEncoderLayer(args.coarse_embed_dim, num_heads,  dim_feedforward = 2 * args.coarse_embed_dim) for _ in range(num_layers)])


        """
        Textual branch
        """
        self.language_encoder = LanguageEncoder(args.coarse_embed_dim,  
                                                hungging_model = args.hungging_model, 
                                                fixed_embedding = args.fixed_embedding, 
                                                intra_module_num_layers = args.intra_module_num_layers, 
                                                intra_module_num_heads = args.intra_module_num_heads, 
                                                is_fine = False,  
                                                inter_module_num_layers = args.inter_module_num_layers,
                                                inter_module_num_heads = args.inter_module_num_heads,
                                                ) 


        print(
            f"CellRetrievalNetwork, class embed {args.class_embed}, color embed {args.color_embed}, dim: {args.coarse_embed_dim}, features: {args.use_features}"
        )


    def encode_text(self, descriptions):

        description_encodings = self.language_encoder(descriptions)  # [B, DIM]

        description_encodings = F.normalize(description_encodings)

        return description_encodings

    def encode_objects(self, objects, object_points):
        """
        Process the objects in a flattened way to allow for the processing of batches with uneven sample counts
        """
        
        batch = []  # Batch tensor to send into PyG
        for i_batch, objects_sample in enumerate(objects):
            for obj in objects_sample:
                # class_idx = self.known_classes.get(obj.label, 0)
                # class_indices.append(class_idx)
                batch.append(i_batch)
        batch = torch.tensor(batch, dtype=torch.long, device=self.device)

        embeddings, pos_postions = self.object_encoder(objects, object_points)

        object_size = self.object_size

        index_list = [0]
        last = 0
        
        x = torch.zeros(len(objects), object_size, self.embed_dim).to(self.device)
      

        for obj in objects:
            index_list.append(last + len(obj))
            last += len(obj)
        
        embeddings = F.normalize(embeddings, dim=-1)  

        for idx in range(len(index_list) - 1):
            num_object_raw = index_list[idx + 1] - index_list[idx]
            start = index_list[idx]
            num_object = num_object_raw if num_object_raw <= object_size else object_size
            x[idx,: num_object] = embeddings[start : (start + num_object)]
            
        
        x = x.permute(1, 0, 2).contiguous()
        for idx in range(len(self.obj_inter_module)):
            x = self.obj_inter_module[idx](x)

        del embeddings, pos_postions
        
        x = x.max(dim = 0)[0]
        x = F.normalize(x)

        return x

    def forward(self):
        raise Exception("Not implemented.")

    @property
    def device(self):
        return self.language_encoder.device

    def get_device(self):
        return self.language_encoder.device

