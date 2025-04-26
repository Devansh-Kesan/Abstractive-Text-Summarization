import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartForConditionalGeneration, BartConfig

class EnhancedBartForSummarization(nn.Module):
    def __init__(self, model_name="facebook/bart-base", use_coverage=True, lambda_coverage=1.0):
        super(EnhancedBartForSummarization, self).__init__()
        self.bart = BartForConditionalGeneration.from_pretrained(model_name)
        self.use_coverage = use_coverage
        self.lambda_coverage = lambda_coverage
        if self.use_coverage:
            # Coverage attention components
            config = self.bart.config
            self.coverage_projection = nn.Linear(1, config.d_model)
            self.coverage_vector = None
    
    @classmethod
    def from_pretrained(cls, model_path, *args, **kwargs):
        """
        Load a model from a pretrained path.
        This method creates a new instance of the class and loads the BART model
        from the specified path.
        """
        # Extract any custom arguments we need
        use_coverage = kwargs.pop('use_coverage', True)
        lambda_coverage = kwargs.pop('lambda_coverage', 1.0)
        
        # Create a new instance
        model = cls(model_name=model_path, use_coverage=use_coverage, lambda_coverage=lambda_coverage)
        
        # Replace the BART model with a loaded version
        model.bart = BartForConditionalGeneration.from_pretrained(model_path, *args, **kwargs)
        
        return model
    
    def save_pretrained(self, output_dir):
        """
        Save the model to a directory.
        """
        # Save the BART model
        self.bart.save_pretrained(output_dir)
        
        # Save custom parameters (if needed)
        # You could save your custom configurations here
        
    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None,
                decoder_attention_mask=None, labels=None):
        if not self.use_coverage:
            # Standard BART forward pass
            outputs = self.bart(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels,
                output_attentions=True,
                return_dict=True
            )
            return outputs
        
        # Enhanced forward pass with coverage mechanism
        batch_size = input_ids.size(0)
        # Initialize coverage vector (batch_size, 1, seq_len)
        seq_len = input_ids.size(1)
        if self.coverage_vector is None or self.coverage_vector.size(0) != batch_size:
            self.coverage_vector = torch.zeros(batch_size, 1, seq_len).to(input_ids.device)
        else:
            self.coverage_vector = self.coverage_vector.detach()
            self.coverage_vector.zero_()
        
        # Get BART model outputs
        outputs = self.bart(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            output_attentions=True,
            return_dict=True
        )
        
        # Extract cross-attentions and update coverage vector
        cross_attentions = outputs.cross_attentions
        # Calculate coverage loss
        coverage_loss = 0
        for layer_attention in cross_attentions:
            # Take average over attention heads
            attn = layer_attention.mean(dim=1)  # (batch_size, tgt_len, src_len)
            # Calculate coverage loss: min(coverage_vector, attention)
            step_coverage_loss = torch.min(self.coverage_vector, attn).sum(dim=-1).mean()
            coverage_loss += step_coverage_loss
            # Update coverage vector
            self.coverage_vector = self.coverage_vector + attn.unsqueeze(1)
        
        # Combine losses
        outputs.loss = outputs.loss + self.lambda_coverage * coverage_loss
        return outputs
    
    def generate(self, input_ids, attention_mask=None, **kwargs):
        # Reset coverage vector for generation
        self.coverage_vector = None
        # Use BART's generate method
        return self.bart.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )