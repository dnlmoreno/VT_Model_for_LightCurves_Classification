
import numpy as np
import torch
import cv2

def get_importance(image, texts, processor, model, device, start_layer=-1, start_layer_text=-1):
    # Process inputs and perform a forward pass to get outputs and attentions
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True).to(device)
    outputs = model(**inputs, output_attentions=True)
    logits_per_image = outputs.logits_per_image

    # Prepare for gradient calculation
    batch_size = logits_per_image.shape[0]
    one_hot_indices = logits_per_image.argmax(dim=1, keepdim=True)
    one_hot = torch.zeros_like(logits_per_image).scatter_(1, one_hot_indices, 1).requires_grad_(True)
    loss = (one_hot.cuda() * logits_per_image).sum()
    model.zero_grad()

    # Image relevance calculations using attention blocks
    image_relevance = calculate_relevance(batch_size, model.vision_model, outputs, loss, 'vision_model_output', start_layer, device)

    return image_relevance, inputs, outputs

def calculate_relevance(batch_size, sub_model, outputs, loss, output_key, start_layer, device):
    attn_blocks = list(dict(sub_model.encoder.layers.named_children()).values())
    if start_layer == -1: 
        start_layer = len(attn_blocks) - 1

    num_tokens = outputs[output_key].attentions[0].shape[-1]
    relevance_matrix = torch.eye(num_tokens, dtype=outputs[output_key].attentions[0].dtype).to(device)
    relevance_matrix = relevance_matrix.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)

    for i, _ in enumerate(attn_blocks):
        if i < start_layer:
            continue
        attn_grad = torch.autograd.grad(loss, [outputs[output_key].attentions[i]], retain_graph=True)[0].detach()
        attn = outputs[output_key].attentions[i].detach()
        attn = attn.reshape(-1, attn.shape[-1], attn.shape[-1])
        attn_grad = attn_grad.reshape(-1, attn_grad.shape[-1], attn_grad.shape[-1])
        cam = attn_grad * attn
        cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=1)
        relevance_matrix += torch.bmm(cam, relevance_matrix)

    return relevance_matrix[:, 0, 1:]

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam