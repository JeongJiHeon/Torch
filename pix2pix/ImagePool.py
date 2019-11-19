import torch

def ImagePool(pool, image,device = torch.device('cpu'), max_size = 50):
    gen_image = image.detach().numpy()
    if len(pool) < max_size:
        pool.append(gen_image)
        return image, True
    else:
        p = random()
        if p > 0.5:
            random_id = randint(0, len(pool)-1)
            temp = pool[random_id]
            pool[random_id] = gen_image
            return torch.Tensor(temp).to(device), True
        else:
            return torch.Tensor(gen_image).to(device), False