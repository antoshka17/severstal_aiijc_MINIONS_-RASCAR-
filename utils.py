from PIL import Image
import cv2
import numpy as np
import torch
import os
import random
from collections import deque
import albumentations as A

device = 'cuda' if torch.cuda.is_available() else 'cpu'
 
def set_random_seed(seed: int = 2222, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = deterministic

def read_image(filepath):
    image = Image.open(filepath)
    image = np.array(image)
    image = cv2.resize(image, (512, 512))
    return image


def read_mask(filepath):
    image = Image.open(filepath).convert('L')
    image = np.array(image).astype(np.float64)
    image /= image.max()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i, j] = 0 if image[i, j] == 0.0 else 1.0
    image = image.astype(np.uint8)
    image = cv2.resize(image, (512, 512))
    return image

def mape(counts, counts_true):
    return 100 / len(counts) * sum([abs(counts_true[i] - counts[i]) / counts_true[i]
                                        for i in range(len(counts))])

def bfs(image):
    n,m = len(image), len(image[0])
    graph = {i: [] for i in range(n*m)}
    
    # creating graph
    for row in range(n):
        for col in range(m):
            if image[row][col] == 1:
                u = row * m + col
                adj = []
                if row != 0: adj.append([row-1, col])
                if col != 0: adj.append([row, col-1])
                if row != n-1: adj.append([row+1, col])
                if col != m-1: adj.append([row, col+1])
                
                for point in adj:
                    y, x = point
                    if image[y][x] == 1:
                        v = y*m + x
                        graph[u].append(v)
                del adj, u
                        
    # doing bfs
    used, comps = [False for i in range(n*m)], []
    for cell in range(n*m):
        row = cell // m
        col = cell % m
        if image[row][col] == 0:
            continue
        if used[cell]:
            continue
            
        visited, degs = [cell], [len(graph[cell])]
        q = deque()
        start = cell
        q.append(start)
    
        while q:
            node = q.popleft()
            for v in graph[node]:
                if not used[v]:
                    q.append(v)
                    used[v] = True
                    visited.append(v)
                    degs.append(len(graph[v]))
                    
                
        comps.append([visited, degs])
        
        del visited, degs, q
    return comps 

def sort_points_clockwise(points):
    points = np.array(points)

    center = np.mean(points, axis=0)

    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    
    sorted_indices = np.argsort(angles)
    sorted_points = points[sorted_indices]

    return sorted_points.tolist()

def testing_model(model, image, use_multitask=False):
    #takes image and returns mask in 0-1 format
    model.eval()
    image = transforms_val(image=image[:, :, :])['image']
    image = image.transpose(2, 0, 1)
    image = torch.tensor(image)
    image = image.unsqueeze(dim=0)
    image = image.to(device)
    with torch.no_grad():
        if use_multitask:
            proba, types = model(image[:, :, :, :])
        else:
            proba = model(image[:, :, :, :])
        proba = proba.detach().cpu().numpy()
        output = proba.round()[0,0,:,:]
        proba = proba[0, 0, :, :]
    return ((output, proba[0, 0, ...], types) if use_multitask else (output, proba))


transforms_val = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

