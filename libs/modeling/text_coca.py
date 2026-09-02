import torch
import torch.nn as nn
import open_clip
from .clip import CLIPModel

def data_split_dir(file, data_split, mode, split_num):
        if mode == 'train':
            file_dir = file.format(mode='train',r1=data_split, r2=100-data_split, r3=split_num)
        else:
            file_dir = file.format(mode='test',r1=data_split, r2=100-data_split, r3=split_num)
        return file_dir

class TextFeatures_coca(nn.Module):
    def __init__(
        self,
        model_path,
        subset_file,
        data_split,
        emb_dim,
        split_num,
        freeze_txt_model=True,
    ):
        super().__init__()
        self.train_classes = self._load_classes(subset_file, data_split, 'train', split_num)
        self.test_classes = self._load_classes(subset_file, data_split, 'test', split_num)

        # Initialize CoCA model
        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name="coca_ViT-L-14", 
            pretrained="mscoco_finetuned_laion2B-s13B-b90k"
        )
        self.model = self.model.float()
        self.tokenizer = open_clip.get_tokenizer("coca_ViT-L-14")
        self.text_projection = nn.Linear(self.model.text.output_dim, emb_dim)
        nn.init.xavier_uniform_(self.text_projection.weight, gain=1)
        nn.init.zeros_(self.text_projection.bias)

        if freeze_txt_model:
            self.model.requires_grad_(False)
            self.text_projection.requires_grad_(True)

    def extract_text_emb(self, cls_name, is_prompt=False):
        if is_prompt:
            train_prompt = self.get_prompt(cls_name)
        else:
            train_prompt = cls_name 

        device = next(iter(self.model.parameters())).device
        
        text = self.tokenizer(train_prompt).to(device)
        
        text_emb = self.model.encode_text(text, normalize=False)
        text_emb = self.text_projection(text_emb)
            
        return text_emb
        
    def txt_read(self, file_path, sort=False):
        with open(file_path, 'r') as f:
            cls_name = [cls_name.strip('\n') for cls_name in f.readlines()]

        if sort:
            cls_name = sorted(cls_name)
        
        split_dict = {cls_name: i for i, cls_name in enumerate(cls_name)}

        return split_dict

    def get_prompt(self, cls_name):
        prompt_cls_name = [f'a video of action {c}' for c in cls_name]
        return prompt_cls_name
    
    def _load_classes(self, subset_file, data_split, mode, split_num):
        file_path = data_split_dir(subset_file, data_split, mode, split_num)
        try:
            with open(file_path, 'r') as f:
                cls_name = [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            raise ValueError(f"No class file: {file_path}")

        return cls_name

    def forward(self, batch_size, mode):
        cls_name = self.train_classes if mode == 'train' else self.test_classes
        text_emb = self.extract_text_emb(cls_name, None) 

        if len(text_emb.size()) == 2:
            text_emb = text_emb.expand(batch_size, -1, -1)
        
        split_num_cls = text_emb.size(1)

        return text_emb, split_num_cls 