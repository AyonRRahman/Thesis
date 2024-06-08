#model parameters 
from models.SFM.DispNetS import DispNetS
from models.SFM.DispNetS_mvit import DispNet_mvit
# from models.DepthAnything.DepthAnything import DepthAnythingSFM
from train_only_depth.udepth_model.udepth import UDepth_SFM
from models.SC_SFM.PoseResNet import PoseResNet 
import torch
import timeit


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def predict_fn(model, input_data):
    return model(input_data)
    
# if args.depth_model == 'dispnet':

model = DispNetS(rmi=False)
print('loading dispnet')

# elif args.depth_model=='dispnet_mvit':

# print('loading dispnet mvit')
# model = DispNet_mvit(rmi=True)

# # elif args.depth_model=='udepth':
# print('loading udepth')
# model = UDepth_SFM(load_pretrained=True, rmi=True)

# model = PoseResNet(18, False)
# print('pose net')

total_param = sum(param.numel() for param in model.parameters())
print(f"total network parameter {total_param/1e6}M")

# 1, 1, 256, 464
print(device)
model.to(device)
x = torch.rand(1, 3, 256, 464).to(device)

print('calculating model avg inference time')
number_of_iterations = 100  # Number of times to run the prediction

# Time the prediction function
time_taken = timeit.timeit(lambda: predict_fn(model, x), number=number_of_iterations)

# Calculate average prediction time per iteration
average_prediction_time = time_taken / number_of_iterations

print(f"avg prediction time: {average_prediction_time}")
print(1/average_prediction_time)

