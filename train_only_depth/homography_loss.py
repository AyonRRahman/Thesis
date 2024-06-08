import torch


def compute_ABC(w_t_c, c_R_w, w_t_chat, chat_R_w, c_n, eye):
    """
    Computes A, B, and C matrix given estimated and ground truth poses
    and normal vector n.
    `w_t_c` and `w_t_chat` must have shape (batch_size, 3, 1).
    `c_R_w` and `chat_R_w` must have shape (batch_size, 3, 3).
    `n` must have shape (3, 1).
    `eye` is the (3, 3) identity matrix on the proper device.
    """
    chat_t_c = chat_R_w @ (w_t_c - w_t_chat)
    chat_R_c = chat_R_w @ c_R_w.transpose(1, 2)

    A = eye - chat_R_c
    C = c_n @ chat_t_c.transpose(1, 2)
    B = C @ A
    A = A @ A.transpose(1, 2)
    B = B + B.transpose(1, 2)
    C = C @ C.transpose(1, 2)

    return A, B, C

class HomographyLoss(torch.nn.Module):
    def __init__(self, xmin=0.025, xmax=0.975, device='cpu'):
        
        super().__init__()

        # `xmin` is the minimum distance of observations in all frames
        xmin = torch.tensor(xmin, dtype=torch.float32, device=device)

        # `xmax` is the maximum distance of observations in all frames
        xmax = torch.tensor(xmax, dtype=torch.float32, device=device)

        # `B_weight` and `C_weight` are the weigths of matrices A and B computed from `xmin` and `xmax`
        self.B_weight = torch.log(xmin / xmax) / (xmax - xmin)
        self.C_weight = xmin * xmax

        # `c_n` is the normal vector of the plane inducing the homographies in the ground-truth camera frame
        self.c_n = torch.tensor([0, 0, -1], dtype=torch.float32, device=device).view(3, 1)

        # `eye` is the (3, 3) identity matrix
        self.eye = torch.eye(3, device=device)

    def compute_ABC(self, w_t_c, c_R_w, w_t_chat, chat_R_w, c_n=self.c_n, eye=self.eye):
        """
        Computes A, B, and C matrix given estimated and ground truth poses
        and normal vector n.
        `w_t_c` and `w_t_chat` must have shape (batch_size, 3, 1).
        `c_R_w` and `chat_R_w` must have shape (batch_size, 3, 3).
        `n` must have shape (3, 1).
        `eye` is the (3, 3) identity matrix on the proper device.
        """
        chat_t_c = chat_R_w @ (w_t_c - w_t_chat)
        chat_R_c = chat_R_w @ c_R_w.transpose(1, 2)

        A = eye - chat_R_c
        C = c_n @ chat_t_c.transpose(1, 2)
        B = C @ A
        A = A @ A.transpose(1, 2)
        B = B + B.transpose(1, 2)
        C = C @ C.transpose(1, 2)

        return A, B, C


    def __call__(self, batch):
        #needs editing
        A, B, C = self.compute_ABC(batch['w_t_c'], batch['c_R_w'], batch['w_t_chat'], batch['chat_R_w'], self.c_n, self.eye)

        error = A + B * self.B_weight + C / self.C_weight
        error = error.diagonal(dim1=1, dim2=2).sum(dim=1).mean()
        return error
