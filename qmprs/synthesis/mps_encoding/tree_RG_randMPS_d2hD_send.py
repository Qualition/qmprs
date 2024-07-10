
'''
################# NEED TO CHANGE THIS TO PYTHON VERSION ########################################
# This function optimizes the fidelity and calculates the error density for a disordered system.
function optm_fp_fid_errden_disorder(qP_list, D, Np, N_round)  
  # Calculate the number of sites in the system
  Ns = 2 * Np;

  # Generate site indices for the MPS
  siteids_dum = siteinds(D, Ns);

  # Initialize the MPS as a product of maximally entangled pairs
  psi_init_dum = init_state_pr_D(Np, D, siteids_dum);

  # Apply projectors to obtain the target MPS state
  psi_targ_dum = targ_qP(psi_init_dum, siteids_dum, qP_list, Np);

  # Initialize a diagonal matrix to represent the pair isometries
  wf = zeros((D, D));
  for idx = 1:D
    wf[idx, idx] = 1 / sqrt(D);
  end

  # Create a list to store the pair isometries as ITensors
  global pair_iso_ls = ITensor[];

  # Populate the list with pair isometries
  for idx = 1:Np
    push!(pair_iso_ls, ITensor(wf, siteids_dum[2 * idx - 1], siteids_dum[2 * idx]));
  end

  # Convert the target MPS into a vector of ITensors
  psi_targ_vec = ITensor[];
  for idx = 1:Ns
    push!(psi_targ_vec, psi_targ_dum[idx]);
  end

  # Initialize lists to store the cost function and trace error
  cost_ls = [];
  tr_err_ls = [];

  # Calculate the initial cost function and trace error
  E = psi_targ_vec[1] * psi_targ_vec[2] * pair_iso_ls[1];
  for idx = 2:Np
    E = E * psi_targ_vec[2 * idx - 1] * psi_targ_vec[2 * idx] * pair_iso_ls[idx];
  end
  cost_func_last = abs(Array(E)[1])^2;
  push!(tr_err_ls, abs(1 - Array(E)[1]));
  push!(cost_ls, cost_func_last);

  # Perform optimization rounds to improve fidelity and reduce error
  for idx_tm = 1:N_round
    idx = 1;
    E_dum = psi_targ_vec[1] * psi_targ_vec[2];
    for idx2 = 2:Np
      E_dum = E_dum * psi_targ_vec[2 * idx2 - 1] * psi_targ_vec[2 * idx2] * pair_iso_ls[idx2];
    end

    # Perform singular value decomposition and update the pair isometries
    U, S, V = svd(E_dum, siteids_dum[2 * idx - 1], siteids_dum[2 * idx]);
    pair_iso_ls[idx] = conj(V) * conj(U) * delta(inds(U)[end], inds(V)[end]);
    cost_func = abs(Array(E_dum * pair_iso_ls[idx])[1])^2;
    push!(cost_ls, cost_func);

    # Update the cost function and trace error for each pair
    for idx = 2:Np
      E_dum = psi_targ_vec[1] * psi_targ_vec[2] * pair_iso_ls[1];
      for idx2 = 2:idx - 1
        E_dum = E_dum * psi_targ_vec[2 * idx2 - 1] * psi_targ_vec[2 * idx2] * pair_iso_ls[idx2];
      end
      idx2 = idx;
      E_dum = E_dum * psi_targ_vec[2 * idx2 - 1] * psi_targ_vec[2 * idx2];
      for idx2 = idx + 1:Np
        E_dum = E_dum * psi_targ_vec[2 * idx2 - 1] * psi_targ_vec[2 * idx2] * pair_iso_ls[idx2];
      end
      U, S, V = svd(E_dum, siteids_dum[2 * idx - 1], siteids_dum[2 * idx]);
      pair_iso_ls[idx] = conj(V) * conj(U) * delta(inds(U)[end], inds(V)[end]);
      ovlp = Array(E_dum * pair_iso_ls[idx])[1];
      cost_func = abs(ovlp)^2; # the fidelity
      push!(tr_err_ls, abs(1 - ovlp));
      push!(cost_ls, cost_func);
    end
  end

  # Calculate the error density based on the final cost function
  err_den = -log(cost_ls[end]) / (Np + 1); # divide by Np+1, meaning the whole block here

  # Return the error density, cost list, trace error list, and pair isometries list
  return err_den, cost_ls, tr_err_ls, pair_iso_ls 
end
###############################################################################################




'''
import numpy as np
from scipy.linalg import qr
import quimb.tensor as qtn
import copy
Np = 10# Amount of injective sites
q = 2 # Bond dimension is D = 2^q
Ns = 2*Np
N_round = 5 # Number of variational sweeping rounds for variational method
N_tm = 2 # Number of MPS computed
mark = 'test' # Some labelling
D = 2**q
Np_raw = (Np+1)*(2**4)-1 # Raw number of pairs, block at most 5 times
Ns_raw = 2*Np_raw # Raw number of sites
def qU_qP_SVD(Qpr):
    U,S,V = np.linalg.svd(Qpr)
    return U*V.T, V*np.diag(S)*V.T

def RandomUnitaryMatrix(N, shape):
    # Create a complex matrix with random entries
    x = (np.random.rand(N, N) + np.random.rand(N, N) * 1j) / np.sqrt(2)

    # Perform QR decomposition on the random matrix
    Q, R = qr(x)

    # Ensure the diagonal of R is positive to make Q a unitary matrix
    diagR = np.sign(np.real(np.diag(R)))
    diagR[diagR == 0] = 1

    # Construct the unitary matrix from Q and the adjusted diagonal
    U = np.dot(Q, np.diag(diagR))

    return U.reshape(shape)

def TtoQ_block(T_size, T_blk_ls, D):
    Q_mid = qtn.Tensor(T_blk_ls[0], inds = ('il','i10','ir_dum1'))
    for i in range(1,T_size-1):
        first_ind = 'ir_dum' + str(i)
        mid_ind = 'i' + str(i)
        last_ind = 'ir_dum' + str(i+1)
        T_it = qtn.Tensor(T_blk_ls[i], inds = (first_ind, mid_ind, last_ind))
        Q_mid = Q_mid@T_it
    T_it = qtn.Tensor(T_blk_ls[T_size-1], inds = (last_ind, 'i' + str(T_size-1), 'ir'))
    Q_mid = Q_mid@T_it
    Q_mid = qtn.Tensor(np.transpose(Q_mid.data.reshape(D,D**2,D), (1,0,2)).reshape(D**2,D**2), inds = ('il','ir'))
    return Q_mid

def init_state_pr_D(np,D, siteinds):
    dim,N = siteinds
    tensors = [qtn.Tensor() for _ in range(N)]
    for i in range(N):
        tensors[i].new_ind(f'k{i}', size = D)
    psi_init = qtn.TensorNetwork(tensors)
    for i in range(Np):
        psi_init[2*i] = qtn.Tensor(np.eye(D),inds = (f'k{i}','lk'))
        psi_init[2*i+1] = qtn.Tensor(np.eye(D)/np.sqrt(D), inds = ('lk', f'k{i}'))

    return psi_init

def targ_qP(psi_init, siteinds, qP_list, Np):
    dim, N = siteinds
    psi_targ = copy.deepcopy(psi_init)
    ########## Needs to be looked at #####################
    orthogonalize(psi_targ,0)
    ######################################################
    qP_it_dum = qtn.Tensor(qP_list[0], inds = ('k0p', 'k0'))
    psi_targ[0] = qP_it_dum@psi_targ[0]
    for i in range(Np-1):
        orthogonalize(psi_targ,2*i+1)
        w = psi_targ[2*i+1]*psi_targ[2*i+2]
        id2i = uniqueinds(psi_targ[2*i+1], psi_targ[2*i+2])
        qP_it_dum = qtn.Tensor(qP_list[i+1], inds = (f'k{2*i+1}p', f'k{2*i+2}p', f'k{2*i+1}',f'k{2*i+2}'))
        w = qP_it_dum @w
        idw = list(w.inds)
        ################## Needs to be looked at ########################
        q, r = np.linalg.qr(w) #qr(w, siteids[2i]', idw[2]);
        #################################################################
        psi_targ[2*i+1] = q
        psi_targ[2*i+2] = r
    qP_it_dum = qtn.Tensor(qP_list[Np],inds = (f'k{Ns-1}p',f'k{Ns}'))
    orthogonalize(psi_targ,Ns-1)
    psi_targ[Ns-1] = qP_it_dum@psi_targ[Ns-1]
    psi_targ = psi_targ/np.linalg.norm(psi_targ)
    ############ Needs to be looked at ###################
    noprime(psi_targ)
    ######################################################
    return orthogonalize(psi_targ,0)
        
def middle_tensors(vec_size, mat_size, shape, indices):
    zero_vec = np.zeros(vec_size)
    zero_vec[0] = 1
    zero_vec = qtn.Tensor(zero_vec, inds = (indices[-1],))
    U = qtn.Tensor(RandomUnitaryMatrix(mat_size, shape), inds = indices)
    T_dum = U@zero_vec
    return T_dum

def Q_left():
    T_l = middle_tensors(2*D, 2*D, (2,D,2*D), ("i_phy","i_r","i_dum")) # shape (2,D) (i_phy, i_r)
    Np_raw_T = Np_raw*2*q
    T_mid_ls = [middle_tensors(2, 2*D, (2,D,D,2),
                               ("i_phy","i_r","i_l","i_dum")).data.reshape(D,2,D)
                for _ in range(Np_raw_T-1)]
    Q_l = T_l.reindex({"i_phy":'i1',"i_r": 'ir_dum'}) # shape (2,D) (i1, ir_dum)
    if q>2:
        T_r_it = qtn.Tensor(T_mid_ls[0],inds = ('ir_dum','i0','ir_dum0'))
        for idx in range(1,q-2):
            first_ind = 'ir_dum' + str(idx)
            mid_ind = 'i' + str(idx)
            last_ind = 'ir_dum' + str(idx+1)
            T_r_it = qtn.Tensor(T_mid_ls[idx], inds = (first_ind,mid_ind,last_ind))
            Q_l = Q_l@T_r_it
    
        mid_ind = 'i' + str(q-2)
        first_ind = 'ir_dum' + str(q-2)
        T_r_it = qtn.Tensor(T_mid_ls[q-2], inds = (first_ind,mid_ind,'i_r'))
        Q_l = Q_l@T_r_it
        Q_l = qtn.Tensor(Q_l.data.reshape(D,D), inds = ('i_l','i_r'))
    
    else:
        T_r_it = qtn.Tensor(T_mid_ls[0], inds = ('ir_dum','i0','i_r'))
        Q_l = Q_l@T_r_it
        Q_l = qtn.Tensor(Q_l.data.reshape(D,D), inds = ('i_l','i_r'))

    return Q_l
Q_r = qtn.Tensor(RandomUnitaryMatrix(D, (D,D)), inds = ("i_phy","i_l"))
Q_l = Q_left()

err_den_ls = [] # List to store error density vectors
tr_err_den_ls = [] # list to store trace error density vectors

Q_mid_ls = []
i_st_0 = q
for idx_mid in range(Np_raw-1):
    T_blk_ls = []
    i_st = i_st_0 + (idx_mid)*2*q-1
    for idx in range(i_st, i_st + 2*q):
        T_blk_ls.append(T_mid_ls[idx])
    Q_mid_ls.append(TtoQ_block(2*q,T_blk_ls,D))

qU_l, qP_l = qU_qP_SVD(Q_l.data)
qU_r, qP_r = qU_qP_SVD(Q_r.data)

# Initialize lists for the unitary and projector parts of the middle blocks
qU_mid_ls = []
qP_mid_ls = []
qP_it_ls = [qP_l]

for idx in range(Np_raw-1):
    qU_mid, qP_mid = qU_qP_SVD(Q_mid_ls[idx].data)
    qU_mid_ls.append(qU_mid)
    qP_mid_ls.append(qP_mid)
    qP_it_ls.append(qP_mid.reshape(D**2,D,D))

qP_it_ls.append(qP_r)
qP_list = [qP_l,qP_mid_ls,qP_r]
psi_init_dum = init_state_pr_D(Np_raw, D, (D, Ns_raw))
psi_targ_dum = targ_qP(psi_init_dum, (D, Ns_raw), qP_list, Np_raw)

err_den, cost_ls, tr_err_ls = optm_fp_fid_errden_disorder(qP_list, D, Np_raw, N_round)

tr_err_den_ls.append(tr_err_ls[-1]/(Np_raw+1))
err_den_ls.append(err_den)

def iter(Np_raw,qP_it_ls, N_round, err_den_ls, tr_err_den_ls):
    Np_raw_2 = int(Np_raw/2 - 1/2)
    Ns_raw_2 = 2*Np_raw_2
    qP_ls_2 = []
    qP_it_ls_2 = []
    tn_l = qP_it_ls[0]
    tn_r = qP_it_ls[1]
    qP_it_dum = qtn.Tensor(tn_l,inds = ('i_l','a'))@qtn.Tensor(tn_r,inds = ('i_r','a','b'))
    qp_dum = qtn.Tensor(qP_it_dum.data.reshape(D**3,D), inds = ('i_l','i_r'))
    _, qP_dum = qU_qP_SVD(qP_dum)
    qP_ls_2.append(qP_dum)
    qP_it_ls_2.append(qP_dum)
    for idx in range(1,Np_raw_2):
        tn_l = qP_it_ls[2*idx]
        tn_r = qP_it_ls[2*idx + 1]
        qP_it_dum = qtn.Tensor(tn_l,inds = ('i_l','a','c'))@qtn.Tensor(tn_r,inds = ('i_r','c','b'))
        qp_dum = qtn.Tensor(qP_it_dum.data.reshape(D**4,D**2), inds = ('i_l','i_r'))
        _,qP_dum = qU_qP_SVD(qP_dum)
        qP_ls_2.append(qP_dum)
        qP_it_ls_2.append(qP_dum.reshape(D**2,D,D))
    
    tn_l = qP_it_ls[2*Np_raw_2]
    tn_r = qP_it_ls[2*Np_raw_2 +1]
    qP_it_dum = qtn.Tensor(tn_l,inds = ('i_l','a','b'))@qtn.Tensor(tn_r,inds = ('i_r','b'))
    qp_dum = qtn.Tensor(qP_it_dum.data.reshape(D**3,D), inds = ('i_l','i_r'))
    _,qP_dum = qU_qP_SVD(qP_dum)
    qP_ls_2.append(qP_dum)
    qP_it_ls_2.append(qP_dum)
    
    psi_init_dum = init_state_pr_D(Np_raw_2, D, (D,Ns_raw_2))
    psi_targ_dum = targ_qP(psi_init_dum, (D, Ns_raw_2), qP_ls_2, Np_raw_2)
    err_den, cost_ls, tr_err_ls = optim_fp_fid_errden_disorder(qP_ls_2, D, Np_raw_2, N_round)
    err_den_ls.append(err_den)
    tr_err_den_ls.append(tr_err_ls[-1]/(Np_raw_2+1))
    return Np_raw_2, qP_it_ls_2, err_den_ls, tr_err_den_ls


Np_raw_2, qP_it_ls_2, err_den_ls, tr_err_den_ls = iter(Np_raw,qP_it_ls, N_round, err_den_ls, tr_err_den_ls)
Np_raw_3, qP_it_ls_3, err_den_ls, tr_err_den_ls = iter(Np_raw_2, qP_it_ls_2, N_round, err_den_ls, tr_err_den_ls)
Np_raw_4, qP_it_ls_4, err_den_ls, tr_err_den_ls = iter(Np_raw_3, qP_it_ls_3, N_round, err_den_ls, tr_err_den_ls)
Np_raw_5, qP_it_ls_5, err_den_ls, tr_err_den_ls = iter(Np_raw_4, qP_it_ls_4, N_round, err_den_ls, tr_err_den_ls)
