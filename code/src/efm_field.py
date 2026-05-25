import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import wandb
import math
import typing as tp
from tqdm import tqdm
from IPython.display import clear_output
from scipy.special import kv as bessel_kv, kve as bessel_kve, gammaln
from scipy.spatial import KDTree

from .ode import get_rk45_sampler_pfgm, LearnedImageODESolver


def _kl_knn_estimate(p: np.ndarray, q: np.ndarray, k: int = 5) -> float:
    """kNN-based estimator of KL(P ∥ Q) from samples.

    Wang, Kulkarni & Verdú (2009) "Divergence Estimation for Multidimensional
    Densities Via k-Nearest-Neighbor Distances", IEEE Trans. Inf. Theory.

    Parameters
    ----------
    p : (n, d)  samples from P  (transported distribution T(X))
    q : (m, d)  samples from Q  (target distribution)
    k : number of nearest neighbours

    Returns
    -------
    KL divergence estimate (float); non-negative in expectation.
    """
    n, d = p.shape
    m    = q.shape[0]

    tree_p = KDTree(p)
    tree_q = KDTree(q)

    # k-th NN distance within P (index 0 is self → take index k)
    r_k = tree_p.query(p, k=k + 1)[0][:, k]          # [n]
    # k-th NN distance from each P sample into Q
    s_k = tree_q.query(p, k=k)[0][:, k - 1]          # [n]

    # KL estimate: d/n * Σ log(s_k / r_k) + log(m / (n-1))
    kl = float(d / n * np.sum(np.log((s_k + 1e-10) / (r_k + 1e-10)))
               + np.log(m / (n - 1)))
    return max(kl, 0.0)  # clip to 0 — estimator can go slightly negative



class EFM:
    
    def __init__(self, config):
        self._config = config # private attribute
         
    @property
    def config(self):
        return self._config
    
    
    @config.setter
    def config(self, config):
        print("You modify the configuration for code running")
        self._config = config
    
    
    def __str__(self):
        return "Electrostatic field matching"
    
    
    def __repr__(self):
        return f"EFM({self._config})"
 


    ######################################
    @staticmethod
    def plotVectorField(*,mesh: torch.Tensor,
                         field:torch.Tensor, 
                         p_samples:torch.Tensor,
                         q_samples:torch.Tensor, **kwargs: tp.Any) -> matplotlib.figure.Figure:
        
        
        
        fig = plt.figure()
        fig.set_figheight(kwargs.get('figheight', 7))
        fig.set_figwidth( kwargs.get('figwidth', 7))

        ax = fig.add_subplot(1,1,1, projection='3d' )
        ax.scatter(p_samples[:,0].cpu(),p_samples[:,1].cpu(),p_samples[:,2].cpu(),
                   color='blue',edgecolor='black',s=80,label=r'$x_{+}  \sim \mathbb{P}(x_{+})$')
        ax.scatter(q_samples[:,0].cpu(),q_samples[:,1].cpu(),q_samples[:,2].cpu(),
                   color='red',edgecolor='black',s=80,label=r'$x_{-}  \sim \mathbb{Q}(x_{-})$')
        ax.quiver(mesh[:, 0].cpu(), mesh[:, 1].cpu(), mesh[:,2].cpu(),
                  field[:, 0].cpu(), field[:, 1].cpu(), field[:,2].cpu(),
                  color='black',length=1, normalize=True)
        ax.set_title(kwargs.get("title", "EFM Ground Truth"))
        ax.legend()
        return fig
    ######################################## 
    
    
    
    
    ########################################
    @staticmethod
    def plotTrajectories(*,traj: torch.Tensor,
                         p_samples: torch.Tensor,
                         q_samples: torch.Tensor,**kwargs: tp.Any):

        fig = plt.figure()
        fig.set_figheight(kwargs.get('figheight', 7))
        fig.set_figwidth( kwargs.get('figwidth', 7))

        ax = fig.add_subplot(1,1,1,  projection='3d' )
        ax.scatter(p_samples[:,0].cpu(),p_samples[:,1].cpu(),p_samples[:,2].cpu(),
                   color='blue',edgecolor='black',s=80, label=r'$x_{+}  \sim \mathbb{P}(x_{+})$')



        traj[-1][:,0] = 6.5
        for jdx in range(len(traj[-1])):

            ax.scatter(traj[-1][jdx,0].cpu(),traj[-1][jdx,1].cpu(),traj[-1][jdx,2].cpu(),
            color='lightgreen',edgecolor='black',zorder=20,label=r'$y \sim  T(x_{+})$' if jdx==0 else None,s=80)

            ax.scatter(q_samples[jdx,0].cpu(),q_samples[jdx,1].cpu(),q_samples[jdx,2].cpu(),
                   color='red',edgecolor='black',s=80, label=r'$x_{-}  \sim \mathbb{Q}(x_{-})$' if jdx==0 else None)

        for idx in range(200,230):
            ax.plot(traj[: ,idx,0].cpu(),
                traj[:, idx,1].cpu(),
                traj[:, idx,2].cpu(),
                color='black',linewidth=0.5, zorder=3);

        # KL divergence KL(T(P) ∥ Q) estimated from samples (spatial coords only)
        transported_np = traj[-1][:, 1:].cpu().contiguous().numpy()   # [n, d-1]
        q_np           = q_samples[:, 1:].cpu().contiguous().numpy()  # [m, d-1]
        kl_val = _kl_knn_estimate(transported_np, q_np,
                                  k=kwargs.get('kl_k', 5))

        base_title = kwargs.get("title", "EFM Ground Truth")
        print(f"{base_title}  |  D_KL(T(P) || Q) = {kl_val:.4f}")
        ax.set_title(f"{base_title}\n$D_{{KL}}(T(P) \\| Q) = {kl_val:.4f}$")
        ax.legend()
        return fig, kl_val
    ########################################


    ########################################
    @staticmethod
    def plotComparison2D(*, traj: torch.Tensor,
                         q_samples: torch.Tensor,
                         **kwargs: tp.Any) -> matplotlib.figure.Figure:
        """
        2D scatter: T(P) vs Q in the (x, y) plane at z = L.

        Parameters
        ----------
        traj      : stacked trajectory tensor [steps, n, D+1]
        q_samples : target samples            [m, D+1]
        kl_val    : (optional kwarg) pre-computed KL value to show in title
        """
        transported = traj[-1][:, 1:3].cpu().contiguous().numpy()   # [n, 2]
        q_xy        = q_samples[:, 1:3].cpu().contiguous().numpy()  # [m, 2]

        fig, ax = plt.subplots(figsize=(kwargs.get('figwidth', 6),
                                        kwargs.get('figheight', 6)))

        ax.scatter(q_xy[:, 0], q_xy[:, 1],
                   c='red', alpha=0.3, s=kwargs.get('s', 8),
                   label=r'$x_{-} \sim Q$  (target)')
        ax.scatter(transported[:, 0], transported[:, 1],
                   c='limegreen', edgecolors='black', linewidths=0.3,
                   alpha=0.8, s=kwargs.get('s', 12),
                   label=r'$T(x_{+})$  (transported)')

        base_title = kwargs.get('title', 'T(P) vs Q  [2D projection at z=L]')
        if 'kl_val' in kwargs:
            base_title += f"\n$D_{{KL}} = {kwargs['kl_val']:.4f}$"
        ax.set_title(base_title)
        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')
        ax.legend()
        ax.set_aspect('equal')
        fig.tight_layout()
        return fig
    ########################################


    ######################################
    @staticmethod
    def plot(x: torch.Tensor, **kwargs: tp.Any ) -> matplotlib.figure.Figure:
        
        fig,ax = plt.subplots(kwargs.get('figsize',5),
                              kwargs.get('figsize',5),
                              figsize=(kwargs.get('figsize',5),
                                       kwargs.get('figsize',5)))
        
        for idx in range(kwargs.get('figsize',5)):
            for jdx in range(kwargs.get('figsize',5)):
                img = x[idx, jdx]
                if img.shape[-1] == 1:
                    ax[idx,jdx].imshow(img.squeeze(-1), cmap='gray')
                else:
                    ax[idx,jdx].imshow(img)
                ax[idx,jdx].set_yticks([])
                ax[idx,jdx].set_xticks([])
        fig.tight_layout(pad=kwargs.get('pad',0.00001))
        return fig
    ######################################


    ######################################
    @staticmethod
    def plota(x: torch.Tensor, **kwargs: tp.Any ) -> matplotlib.figure.Figure:
        
        fig,ax = plt.subplots(kwargs.get('figsize',5),
                              kwargs.get('figsize',5),
                              figsize=(kwargs.get('figsize',5),
                                       kwargs.get('figsize',5)))
        
        for idx in range(kwargs.get('figsize',5)):
            for jdx in range(kwargs.get('figsize',5)):
                img = x[idx, jdx].permute(1, 2, 0).cpu()
                if img.shape[-1] == 1:
                    ax[idx,jdx].imshow(img.squeeze(-1), cmap='gray')
                else:
                    ax[idx,jdx].imshow(img)
                ax[idx,jdx].set_yticks([])
                ax[idx,jdx].set_xticks([])
        fig.tight_layout(pad=kwargs.get('pad',0.00001))
        return fig
    ######################################
    
    
    
    ######################################
    @staticmethod
    def plot_trajectory(traj: torch.Tensor, **kwargs: tp.Any) -> matplotlib.figure.Figure:
    
        fig,ax = plt.subplots(kwargs.get('figsize',5),
                              len(traj),
                              figsize=(len(traj),kwargs.get('figsize',5)),
                              sharex=True,sharey=True)
        
        for time in range(len(traj)):
            for idx in range(kwargs.get('figsize',5)):
                img = np.clip(traj[time, idx].permute(1, 2, 0).cpu().numpy() * 255, 0, 255).astype(np.uint8)
                if img.shape[-1] == 1:
                    ax[idx,time].imshow(img.squeeze(-1), cmap='gray')
                else:
                    ax[idx,time].imshow(img)
                ax[idx,time].set_xticks([])
                ax[idx,time].set_yticks([])

        fig.tight_layout(pad=kwargs.get('pad',0.00001))
        return fig
    ######################################
    
    
    
    
    ########################################
    @staticmethod
    def get_mesh( kv : tp.Dict[str, float], 
                  mesh_num_points: tp.Optional[int] = None) -> torch.Tensor:
  
        num_points = 10 if mesh_num_points is None else mesh_num_points
    
        if len(kv) == 3:
            mesh = []
            z = torch.linspace(kv["z"][0],kv["z"][1], mesh_num_points)
            x = torch.linspace(kv["x"][0], kv["x"][1], mesh_num_points)
            y = torch.linspace(kv["y"][0], kv["y"][1], mesh_num_points)

            for i in x:
                for j in y:
                    for k in z:
                        mesh.append(torch.tensor([i,j,k]))
        else:
            raise ValueError("The length of DICT MESH should be 3!")
        return torch.stack(mesh, dim=0)
    ########################################
    
   
    
    
    ########################################
    def _compute_field(self,
                       perturbed_samples_vec: torch.Tensor,
                       p_samples: torch.Tensor,
                       q_samples: torch.Tensor) -> torch.Tensor:
        m = getattr(self._config.training, 'm', 0.0)
        if m > 0.0:
            return self.GroundTruthYukawa(perturbed_samples_vec, p_samples, q_samples, m=m)
        return self.GroundTruth(perturbed_samples_vec, p_samples, q_samples)
    ########################################


    ########################################
    def Toytrain(self,  p_dist ,
                    q_dist ,
                    net: tp.Callable[[torch.Tensor], torch.Tensor],
                    optimizer, **kwargs: tp.Any ): #-> tp.Sequence[tp.Callable[[torch.Tensor], torch.Tensor], tp.Sequence[int]]:
        
        
        losses = []
        for step in tqdm(range(self._config.training.training_steps)):

            optimizer.zero_grad()
            p_samples =  p_dist.sample(self._config.training.batch_size).to(self._config.device) #[B,D]
            q_samples =  q_dist.sample(self._config.training.batch_size).to(self._config.device) #[B,D]

            perturbed_samples_vec = self.forward_interpolation(p_samples, q_samples)

            field = self._compute_field(perturbed_samples_vec, p_samples.clone(),
                                                               q_samples.clone())
            
            #field = math.sqrt(self._config.DIM)*field/( torch.norm(field, dim=1, keepdim=True) + 1e-5)
            pred  = net(perturbed_samples_vec)
            loss = torch.mean((field - pred)**2)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
            if kwargs.get("verbose",False):
                clear_output(wait=True)
                plt.plot(losses)
                plt.show()
            
        return net, losses
    ######################################## 
    
    
    
    #######################################
    def train(self, train_loader, eval_loader, net, optimizer, optimize_fn,
                    state,  **kwargs: tp.Any):
        
       
        train_iter = iter(train_loader)
        eval_iter = iter(eval_loader)
        
        for step in tqdm(range(self._config.training.n_iters  + 1)):
            
            
            ###############################
            try:
                batch_x,_ =  next(train_iter)
            except StopIteration:
                print('stop')
            else:
                train_iter = iter(train_loader)
                batch_x,_ =  next(train_iter)
            batch_x = batch_x.to(self._config.device)
            batch_y = torch.randn_like(batch_x).to(self._config.device)
            ###############################
            
            
            optimizer = state['optimizer']
            optimizer.zero_grad()
            
            
            ###############################
            perturbed_samples_vec = self.forward_interpolation(batch_x[:self._config.training.small_batch_size],
                                                               batch_y[:self._config.training.small_batch_size])
            
            if torch.isnan(perturbed_samples_vec).any():
                print('NaN in perturbed samples — resampling')
                perturbed_samples_vec = self.forward_interpolation(batch_x[:self._config.training.small_batch_size],
                                                               batch_y[:self._config.training.small_batch_size])
            ###############################



            ###############################
            field = self._compute_field(perturbed_samples_vec,
                                        torch.cat([self._config.p.x_loc*torch.ones(len(batch_x))[:,None].to(self._config.device),
                                                     batch_x.view(-1,self._config.DIM-1)], dim=1),
                                        torch.cat([self._config.q.x_loc*torch.ones(len(batch_y))[:,None].to(self._config.device),
                                                     batch_y.view(-1,self._config.DIM-1)], dim=1))

            if torch.isnan(field).any():
                print('NaN in Ground Truth field')
            ###############################


            #field = math.sqrt(self._config.DIM)*field/( torch.norm(field, dim=1, keepdim=True) + 1e-5)


            perturbed_samples_x = perturbed_samples_vec[:, 1:].view(-1,self._config.data.num_channels,
                                                                    self._config.data.image_size,
                                                                    self._config.data.image_size)
            perturbed_samples_z = perturbed_samples_vec[:, 0]
            net_x, net_z = net(perturbed_samples_x, perturbed_samples_z)

            if torch.isnan(net_x).any() or torch.isnan(net_z).any():
                print('NaN in network prediction')
                
            net_x = net_x.view(net_x.shape[0], -1)
            # Predicted N+1-dimensional Poisson field
            pred = torch.cat([net_z[:, None], net_x], dim=1)
             
            loss = torch.mean((field - pred)**2)
 
            loss.backward()
            optimize_fn(optimizer, net , step=state['step'], config=self._config)
            state['step'] += 1
            state['ema'].update(net.parameters())
            if step % 10 == 0:
                wandb.log({"loss train": loss.item()}, step=step)
            
            
            if step % self._config.training.eval_freq == 0:
        
                try:
                    batch_x,_ =  next(eval_iter)
                except StopIteration:
                    print('stop')
                else:
                    eval_iter = iter(eval_loader)
                    batch_x,_ =  next(eval_iter)
                batch_x = batch_x.to(self._config.device) 
                batch_y = torch.randn_like(batch_x).to(self._config.device)


                with torch.no_grad():
                    ema = state['ema']
                    ema.store(net.parameters())
                    ema.copy_to(net.parameters())
                    
                    perturbed_samples_vec = self.forward_interpolation(batch_x[:self._config.training.small_batch_size],
                                                                       batch_y[:self._config.training.small_batch_size])
            
                    field = self._compute_field(perturbed_samples_vec,
                                               torch.cat([self._config.p.x_loc*\
                                                            torch.ones(len(batch_x))[:,None].to(self._config.device),
                                                            batch_x.view(-1,self._config.DIM-1)], dim=1),
                                               torch.cat([self._config.q.x_loc*\
                                                            torch.ones(len(batch_y))[:,None].to(self._config.device),
                                                            batch_y.view(-1,self._config.DIM-1)], dim=1))
                    
                    #field = field/( torch.norm(field, dim=1, keepdim=True) + 1e-5) 
                    perturbed_samples_x = perturbed_samples_vec[:, 1:].view(-1,self._config.data.num_channels,
                                                                    self._config.data.image_size,
                                                                    self._config.data.image_size)
                    perturbed_samples_z = perturbed_samples_vec[:, 0]
                    net_x, net_z = net(perturbed_samples_x, perturbed_samples_z)
                    net_x = net_x.view(net_x.shape[0], -1)
                    # Predicted N+1-dimensional Poisson field
                    pred = torch.cat([net_z[:, None], net_x], dim=1)

                    eval_loss = torch.mean((field - pred)**2)
   
                    ema.restore(net.parameters())
                    wandb.log({"loss eval":eval_loss.item()},step=step)
                    
            
            # sampling #
            
            if step % self._config.training.snapshot_freq == 0:
                
                with torch.no_grad():
                    ema.store(net.parameters())
                    ema.copy_to(net.parameters())

                    shape = (25, self._config.data.num_channels,
                                 self._config.data.image_size, self._config.data.image_size)

                    batch_y = torch.randn(*shape)
                    
                    # first sampling procedure #
                    sampling_fn = get_rk45_sampler_pfgm(y=batch_y , config=self._config,
                                                       shape=shape,
                                                       eps=self._config.training.epsilon,
                                                       device=self._config.device)
                    sample, n, traj = sampling_fn(net, batch_y)
                    # first sampling procedure #
                    
                    sample = np.clip(sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
                    batch_y = np.clip(batch_y.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
                    C, H = self._config.data.num_channels, self._config.data.image_size
                    fig_1 = self.plot(sample.reshape(5, 5, H, H, C))
                    fig_2 = self.plot(batch_y.reshape(5, 5, H, H, C))
                    fig_3 = self.plot_trajectory(traj)
                    wandb.log({"Generated Images RK45":fig_1},step=step)
                    wandb.log({"Init Images":fig_2},step=step)
                    wandb.log({"Trajectories RK45":fig_3},step=step)


                    
                    # second sampling procedure #
                    ode_solver = LearnedImageODESolver(net , self._config)
                    batch_y = torch.randn(*shape).to(self._config.device)
                    sample, traj = ode_solver(torch.cat([(self._config.L)*torch.ones(batch_y.shape[0],
                                                                                    device=batch_y.device)[:,None],
                                                         batch_y.view(-1, self._config.DIM-1)],dim=1).to(self._config.device))
                    # second sampling procedure #
                    
                    #sample = np.clip(sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
                    #batch_y = np.clip(batch_y.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
                     
                    fig_1 = self.plota(sample[:,1:].reshape(5, 5, C, H, H).detach().cpu())
                    #fig_3 = self.plot_trajectory(traj)
                    wandb.log({"Generated Images Euler":fig_1},step=step)
                    #wandb.log({"Trajectories Euler":fig_3},step=step)
    
        return net, state
    #######################################
    



    
    ######################################## 
    def forward_interpolation(self,
                              p_samples: torch.Tensor,
                              q_samples: torch.Tensor) -> torch.Tensor:
        
        """
        The definition Inter-plate points between plates.
        
        Input:
        p_samples - torch.Size([b,D+1]) or torch.Size([b,C,H,W])
        q_samples - torch.size([b,D+1]) or torch.Size([b,C,H,W])
        
        Return:
        perturbed_vec_samples - torch.Size([B,D+1])
        """
        
        

        #################################################
        ######       Mesh Interpolation (Toy)      ######
        #################################################
        if self._config.training.interpolation == 'mesh':
            
            mesh = self.get_mesh(self._config.KV, self._config.mesh_num_points).to(self._config.device)
            idxs = torch.randperm(mesh.shape[0])
            perturbed_samples_vec = mesh[idxs][:self._config.training.small_batch_size]
        #################################################
        ######   Mesh Interpolation (Toy)          ######
        #################################################   
        

        
        
        #################################################
        ######   Uniform Interpolation             ######
        ################################################# 
        elif self._config.training.interpolation == 'Uniform':

            t = torch.distributions.Uniform(low=self._config.p.x_loc + self._config.training.epsilon,
                high=self._config.q.x_loc - self._config.training.epsilon).sample(\
                torch.Size([self._config.training.small_batch_size])).to(self._config.device)[:, None] #[b,1]
            
            den = self._config.L - 2*self._config.training.epsilon
            
            """
            perturbed_x = (t-self._config.training.epsilon)/den*\
                          q_samples[:self._config.training.small_batch_size,1:]\
               +(1-(t-self._config.training.epsilon)/den)*\
               p_samples[:self._config.training.small_batch_size,1:] #[b, D+1]
            """
            
            perturbed_x = q_samples*(t[:,None,None]/self._config.L) + (1 - t[:,None,None]/self._config.L)*p_samples
            perturbed_samples_vec = torch.cat([t,
                                               perturbed_x.reshape(len(p_samples), self._config.DIM-1)], dim=1)
            
            #perturbed_samples_vec = torch.cat([t, perturbed_x], dim=1) 
        #################################################
        ######   Uniform Interpolation             ######
        ################################################# 



        
        #################################################
        ######           PFGM Interpolation        ######
        #################################################
        elif self._config.training.interpolation == 'Gaussian_mixing':

            assert p_samples.shape == torch.Size([self._config.training.small_batch_size, self._config.data.num_channels,
                                                  self._config.data.image_size, self._config.data.image_size]) 
            assert p_samples.shape == q_samples.shape
            
            ####### perturbation for z component #######
            m = torch.rand((p_samples.shape[0],), device=p_samples.device) * self._config.training.M  #[b] : m ~ U[0,M]
            tau = self._config.training.tau 
            z = torch.randn( p_samples.shape[0], 1).to(p_samples.device)*self._config.training.sigma_end #[b, 1] : influence on the performance ??
            z = z.abs()  # [b,1]  : z = epsilon + N(0,I)
            assert z.shape==torch.Size([self._config.training.small_batch_size, 1])
                                      
            # confine norms
            # see Appendix B.1.1 of https://arxiv.org/abs/2209.11178
            """
            if config.training.restrict_M:
                idx = (z < self._config.training.epsilon + 0.005).squeeze()
                num = int(idx.int().sum())
                restrict_m = int(self._config.training.M * 0.7)
                m[idx] = torch.rand((num,), device=p_samples.device) * restrict_m
            """
            if self._config.training.restrict_M:
                idx = (z < 0.005).squeeze()
                num = int(idx.int().sum())
                restrict_m = int(self._config.training.M * 0.7)
                m[idx] = torch.rand((num,), device=p_samples.device) * restrict_m
            
 
            multiplier = (1+tau) ** m # torch.Size([b]) : the essence of this form??
            perturbed_z = z.squeeze() * multiplier # torch.Size([b])* torch.Size([b]) = torch.Size([b])
            """
            perturbed_z = torch.clamp(perturbed_z, min=self._config.training.epsilon ,
                                                   max=self._config.L - self._config.training.epsilon) #torch.Size([b])
            """
            ####### perturbation for z component #######
            

            ####### perturbation for x component #######
            # Sample uniform angle
            gaussian = torch.randn(p_samples.shape[0], self._config.DIM-1).to(p_samples.device) # torch.Size([b, C*H*W=D])
            unit_gaussian = gaussian / torch.norm(gaussian, p=2, dim=1, keepdim=True) #  torch.Size([b, C*H*W=D])
            noise = torch.randn_like(p_samples).reshape(p_samples.shape[0] , 
                                                        -1) * self._config.training.sigma_end #torch.Size([b, C*H*W=D])
            norm_m = torch.norm(noise, p=2, dim=1) * multiplier # torch.Size([b])*torch.Size([b]) = torch.Size([b])

            # Construct the perturbation for x
            perturbation_x = unit_gaussian * norm_m[:, None] # torch.Size([b,C*H*W])* torch.Size([b,1])=  torch.Size([b,C*H*W=D])
            perturbation_x = perturbation_x.view_as(p_samples) # torch.size([b,C,H,W])
            perturbed_x = p_samples + perturbation_x  # torch.size([b,C,H,W])  + torch.size([b,C,H,W])
            ####### perturbation for x component #######
            
            perturbed_samples_vec = torch.cat((perturbed_z[:, None],
                                               perturbed_x.reshape(p_samples.shape[0], self._config.DIM - 1),
                                               ), dim=1) #[b, D+1]
        #################################################
        ######           PFGM Interpolation        ######
        #################################################

        elif self._config.training.interpolation == 'Uniform_mixing':
            
            
            assert p_samples.shape == torch.Size([self._config.training.small_batch_size, self._config.data.num_channels,
                                                  self._config.data.image_size, self._config.data.image_size]) 
            assert p_samples.shape == q_samples.shape
            
            ####### perturbation for z component #######
            m = torch.rand((p_samples.shape[0],), device=p_samples.device) * self._config.training.M  #[b] : m ~ U[0,M]
            tau = self._config.training.tau 
            z = torch.randn( p_samples.shape[0], 1).to(p_samples.device)*self._config.training.sigma_end #[b, 1] : influence on the performance ??
            z = z.abs()  # [b,1]  : z = epsilon + N(0,I)
            assert z.shape==torch.Size([self._config.training.small_batch_size, 1])
                                      
            # confine norms
            # see Appendix B.1.1 of https://arxiv.org/abs/2209.11178
            """
            if config.training.restrict_M:
                idx = (z < self._config.training.epsilon + 0.005).squeeze()
                num = int(idx.int().sum())
                restrict_m = int(self._config.training.M * 0.7)
                m[idx] = torch.rand((num,), device=p_samples.device) * restrict_m
            """
            if self._config.training.restrict_M:
                idx = (z < 0.005).squeeze()
                num = int(idx.int().sum())
                restrict_m = int(self._config.training.M * 0.7)
                m[idx] = torch.rand((num,), device=p_samples.device) * restrict_m
            
 
            multiplier = (1+tau) ** m # torch.Size([b]) : the essence of this form??
            perturbed_z = z.squeeze() * multiplier # torch.Size([b])* torch.Size([b]) = torch.Size([b])
            
            
            perturbed_x = q_samples*(perturbed_z[:,None,None,None]/self._config.L) + (1 - perturbed_z[:,None,None,None]/self._config.L)*p_samples
            perturbed_samples_vec = torch.cat([perturbed_z[:, None],
                                               perturbed_x.reshape(len(p_samples), self._config.DIM-1)], dim=1)
            
        elif self._config.training.interpolation == 'both_side':
            pass
    
        return perturbed_samples_vec
    ########################################        
                     





        
        
    ########################################                                   
    def GroundTruth(self, 
                    perturbed_samples_vec: torch.Tensor,
                    p_samples: torch.Tensor,
                    q_samples: torch.Tensor,
                    **kwargs: tp.Any) -> torch.Tensor:
        """
        input:

        perturbed_samples_vec - torch.Size([b,D+1]) 
        p_samples - torch.Size([B,D+1])
        q_samples - torch.Size([B,D+1])
        config

        output:

        Superposition field - torch.Size([B, D+1])

        source: https://github.com/Newbeeer/Poisson_flow/blob/main/losses.py
        """
                                      
        gt_distance_x = torch.norm((perturbed_samples_vec.unsqueeze(1) - p_samples),dim=-1) # [b,B]
        gt_distance_y = torch.norm((perturbed_samples_vec.unsqueeze(1) - q_samples),dim=-1) # [b,B]
         
        #assert gt_distance_x.shape==torch.Size([self._config.training.small_batch_size,self._config.training.batch_size])
    

        # For numerical stability, timing each row by its minimum value
        if self._config.training.stability:
            distance_x = torch.min(gt_distance_x, dim=1, keepdim=True)[0] / (gt_distance_x + 1e-7) #[b,1]/[b,B] = [b,B]
            distance_y = torch.min(gt_distance_y, dim=1, keepdim=True)[0] / (gt_distance_y + 1e-7) #[b,1]/[b,B] = [b,B]
        else:
            distance_x = 1./ (gt_distance_x + 1e-7) # [b,B]
            distance_y = 1./ (gt_distance_y + 1e-7) # [b,B]

        data_dim = self._config.DIM # N+1
        distance_x = distance_x ** data_dim # [b,B]
        distance_y = distance_y ** data_dim # [b,B]


        distance_x = distance_x[:, :, None] # [b,B,1]
        distance_y = distance_y[:, :, None] # [b,B,1]

        # Normalize the coefficients (effectively multiply by c(\tilde{x}) in the paper)

        coeff_x = distance_x / (torch.sum(distance_x, dim=1, keepdim=True)  ) # [b,B,1]
        coeff_y = distance_y / (torch.sum(distance_y, dim=1, keepdim=True)  ) # [b,B,1]

        diff_x = - (perturbed_samples_vec.unsqueeze(1) - p_samples) # [b,B,D+1]
        diff_y = - (perturbed_samples_vec.unsqueeze(1) - q_samples) # [b,B,D+1]

        # Calculate empirical Poisson field (N+1 dimension in the augmented space)
        gt_direction_x = torch.sum(coeff_x * diff_x, dim=1) #[b,D+1]
        gt_direction_y = torch.sum(coeff_y * diff_y, dim=1) #[b,D+1]
        assert len(gt_direction_x.shape)==2
        assert gt_direction_x.shape[1]==self._config.DIM 


        gt_direction_x = gt_direction_x.view(gt_direction_x.size(0), -1)#[b,D+1]
        gt_direction_y = gt_direction_y.view(gt_direction_y.size(0), -1)#[b,D+1]

        
        # Normalizing the N+1-dimensional Poisson field
        gt_norm_x = gt_direction_x.norm(p=2, dim=-1)
        gt_norm_y = gt_direction_y.norm(p=2, dim=-1)

        #if kwargs.get('Normalized',True):
        if True:
            gt_direction_x /= (gt_norm_x.view(-1, 1) + self._config.training.gamma)
            gt_direction_y /= (gt_norm_y.view(-1, 1) + self._config.training.gamma)

        
        gt_direction_x *= np.sqrt(self._config.DIM)
        gt_direction_y *= np.sqrt(self._config.DIM)
        
        
        return  - gt_direction_x +  gt_direction_y
    ########################################


    ########################################
    def GroundTruthYukawa(self,
                          perturbed_samples_vec: torch.Tensor,
                          p_samples: torch.Tensor,
                          q_samples: torch.Tensor,
                          m: float,
                          **kwargs: tp.Any) -> torch.Tensor:
        """
        Yukawa (screened-Poisson) ground-truth field.

        Replaces the Poisson kernel  1/ρ^{N+1}  with the Yukawa kernel
            w(ρ) = K_{ν}(m·ρ) / ρ^ν,   ν = DIM/2,   DIM = N+1
        derived from the augmented-space Green's function (eq. 52 in the paper):
            G(x,t;x') ∝ (m/ρ)^{(N-1)/2} · K_{(N-1)/2}(m·ρ)
        Taking −∇G and using  d/dx[x^{−ν}K_ν(x)] = −x^{−ν}K_{ν+1}(x)  gives the
        scalar kernel weight  K_{(N+1)/2}(m·ρ) / ρ^{(N+1)/2} = K_{ν}(m·ρ) / ρ^ν.

        When m → 0 :  K_ν(x) ~ Γ(ν)/2 · (2/x)^ν  →  w ~ 1/ρ^{2ν} = 1/ρ^{DIM}
        recovering the Poisson kernel exactly.

        Parameters
        ----------
        perturbed_samples_vec : [b, D+1]
        p_samples             : [B, D+1]
        q_samples             : [B, D+1]
        m                     : Yukawa screening mass (m > 0)
        """
        nu    = self._config.DIM / 2.0
        gamma = self._config.training.gamma

        rho_x = torch.norm(perturbed_samples_vec.unsqueeze(1) - p_samples, dim=-1)  # [b, B]
        rho_y = torch.norm(perturbed_samples_vec.unsqueeze(1) - q_samples, dim=-1)  # [b, B]

        rho_x_np = (rho_x + 1e-7).detach().cpu().numpy()
        rho_y_np = (rho_y + 1e-7).detach().cpu().numpy()

        # Compute log(K_ν(m·ρ) / ρ^ν) stably to avoid overflow when ν = DIM/2 is large.
        # For m·ρ < ν: K_ν(x) ≈ Γ(ν)/2·(2/x)^ν  →  log K_ν(x) ≈ gammaln(ν)+(ν-1)·log2 − ν·log x
        # For m·ρ ≥ ν: use kve(ν,x) = K_ν(x)·exp(x)  →  log K_ν(x) = log(kve(ν,x)) − x
        # Then apply the log-softmax trick (subtract row max) before exp.
        def _log_yukawa_weight(rho: np.ndarray) -> np.ndarray:
            z = m * rho                                      # [b, B]
            log_kv = np.empty_like(rho)
            small = z < nu
            if small.any():
                log_kv[small] = (gammaln(nu) + (nu - 1) * np.log(2)
                                 - nu * np.log(np.maximum(z[small], 1e-300)))
            if (~small).any():
                log_kv[~small] = (np.log(np.maximum(bessel_kve(nu, z[~small]), 1e-300))
                                  - z[~small])
            return log_kv - nu * np.log(rho)

        log_w_x = _log_yukawa_weight(rho_x_np)              # [b, B]
        log_w_y = _log_yukawa_weight(rho_y_np)
        w_x = np.exp(log_w_x - log_w_x.max(axis=1, keepdims=True))  # log-softmax trick
        w_y = np.exp(log_w_y - log_w_y.max(axis=1, keepdims=True))

        device = perturbed_samples_vec.device
        dtype  = perturbed_samples_vec.dtype
        distance_x = torch.tensor(w_x, device=device, dtype=dtype).unsqueeze(-1)  # [b, B, 1]
        distance_y = torch.tensor(w_y, device=device, dtype=dtype).unsqueeze(-1)

        coeff_x = distance_x / torch.sum(distance_x, dim=1, keepdim=True)  # [b, B, 1]
        coeff_y = distance_y / torch.sum(distance_y, dim=1, keepdim=True)

        diff_x = -(perturbed_samples_vec.unsqueeze(1) - p_samples)  # [b, B, D+1]
        diff_y = -(perturbed_samples_vec.unsqueeze(1) - q_samples)

        gt_direction_x = torch.sum(coeff_x * diff_x, dim=1)  # [b, D+1]
        gt_direction_y = torch.sum(coeff_y * diff_y, dim=1)

        gt_direction_x /= (gt_direction_x.norm(p=2, dim=-1).view(-1, 1) + gamma)
        gt_direction_y /= (gt_direction_y.norm(p=2, dim=-1).view(-1, 1) + gamma)

        gt_direction_x *= np.sqrt(self._config.DIM)
        gt_direction_y *= np.sqrt(self._config.DIM)

        return -gt_direction_x + gt_direction_y
    ########################################
