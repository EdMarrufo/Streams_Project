import numpy as np
import matplotlib.pyplot as plt
from cosmodm.wdm_model import WDMModel
from cosmodm.cdm_model import CDMModel
from cosmodm.fdm_model import FDMModel, FDMDu2018
from scipy import interpolate
import matplotlib.ticker as mticker
import scipy
import matplotlib.ticker as mticker
import astropy.units as u

# Constants 

G=4.30091e-3*u.pc*u.Msun**-1 * (u.km/u.s)**2
tmax=3.4 # Age of Pal5 stream
sigma=180
r_stream=13 # Pal5 stream modeled in a circular orbit 


######################## Mass PDF for sampling (including plotting function) ##############################################

def dN_dM_lcdm_Erkal_2016(mmin,mmax):
    m0=2.52e7
    a0=3.26e-5
    n=-1.9
    steps=int(1e6)
    M=np.linspace(mmin,mmax,1000000)
    return M**(n) * (1/(m0)**n) *a0, M

# Uses the form of LCDM SHMF defined in Erkal et al. 2016 
# Inverse CDF method to sample halo mass 

def draw_mass_lcdm_Erkal_2016(Mmin,Mmax,n):
    M=np.linspace(Mmin,Mmax,1000000+1)
    dx= (Mmax - Mmin) /1.e6
    CDF = np.insert(np.cumsum(dx * dN_dM_lcdm_Erkal_2016(Mmin,Mmax)[0]), 0, 0.)
    CDF = CDF /CDF[-1]
    solve = scipy.interpolate.interp1d(CDF, M)
    return solve(np.random.uniform(size=n))

# Uses inverse CDF method to sample halo mass values from pdf, i.e. SHMFs from CosmoDM DM models
def sample_mass(model,mmin,mmax,size=None):
    steps=int(1e6)
    mhalo=np.linspace(mmin,mmax,steps)
    dx=(mmax - mmin)/steps
    pdf=model.subhalo_mass_function(mhalo)/mhalo # dN/dlogM -> dN/dM
    cdf=np.insert(np.cumsum(dx * pdf), 0, 0.)
    cdf=cdf/cdf[-1]
    inverse_cdf=interpolate.interp1d(cdf[:-1], mhalo,fill_value="extrapolate")
    
    # unnormalized mass draws 
    mass_draws=inverse_cdf(np.random.uniform(size=size))
    
    # normalization factor 
    
    integral=float(np.trapz(pdf,x=mhalo)) 
    norm_factor=integral/size
    
    return mass_draws, norm_factor 

def plot_mass_sampling(mass_arr,model):
    f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
    M=np.linspace(10**5,10**9,1000000)
    bins=np.linspace(10**5,10**9, 20000)
    fig, ax = plt.subplots(figsize=(11,11))
    plt.plot(M,CDMModel().subhalo_mass_function(M)/M, color='red',linewidth=3, linestyle='--',label="CDM")
    colormap = plt.cm.Paired
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 2, 30))))

   
    for idx,mass_val in enumerate(mass_arr):
        m_draw, norm = sample_mass(model(mass=mass_val),10**5,10**9,size=100000000)
        n,x=np.histogram(m_draw,bins=bins)
        bin_centers = 0.5*(x[1:]+x[:-1])
        diff_bins=np.diff(bins)
        plt.plot(M,model(mass=mass_val).subhalo_mass_function(M)/M,linewidth=3)
        plt.plot(bin_centers,(n*norm)/diff_bins,linestyle='--',linewidth=5,label="$FDMm={}eV$".format(f._formatSciNotation('%1.10e' % (mass_val*1e-22)))) #label="FDMm={:.1f}KeV".format(mass_val)
    
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel("dN/dM", fontsize=30)
    plt.xlabel("$M_{\odot}$",fontsize=30)
    plt.legend(fontsize=15,loc='upper right')
    plt.tick_params(labelsize=16)
    plt.grid()
    plt.show()

######################################################################################################################################################

######################################### Gap Size Distribution ######################################################################################


def w_per(v_dispersion):
    wp=np.linspace(-1000,1000,100000)
    p_per=np.sqrt(2/np.pi) * wp**2 * v_dispersion**-3 * np.exp(-wp**2/(2*v_dispersion**2))
    return np.array(p_per)

def get_impact_times(time_size):
    return np.random.triangular(0,0,tmax, size=time_size)

def assign_velocities(n):
    
    # Aassign w_parallel values 
    assign_w_parallel=np.random.normal(-220.,180.,n)
    
    # Assign perpendicular velocity
    vals=np.linspace(-1000,1000,100000)
    pdf_wperp=w_per(180.)
    pdf_wperp/=pdf_wperp.sum()
    assign_wperp=np.random.choice(vals, size=n, p=pdf_wperp)
    
    # Assign omega values
    w_values=(assign_w_parallel**2. + assign_wperp**2.)**0.5
    
    return np.array(w_values),np.array(assign_wperp)

def assign_bs(bmax,n):
    b_s=bmax*np.random.uniform(0.,1.,n)
    return b_s

def time_caustic(M, rs, t, r, w, wperp, b):
    tc = 4*w**3/wperp**2 * (b**2 + rs**2)/(G*M)
    tc=tc.decompose()
    return tc.to(u.Gyr)

        
def gap_size(M, rs, t, r, w, wperp, b,t_c):
    
    brs = np.sqrt(b**2 + rs**2)
    
    delta_nonc = 2*w/wperp * brs/r + 2*G*M*wperp*t/(w**2*r*brs)
    delta_c=4 * np.sqrt(2*G*M*t/(w*r**2))
    
    mask=np.where(t.value > t_c.value)
    
    np.put(delta_nonc,mask[0],delta_c[mask])
    
    return (((delta_nonc.decompose()*u.radian).to(u.deg)).value)

def gap_depth(M, rs, t, r, w, wperp, b):
    f = (1 + (wperp**2/w**3 * 2*G*M*t/(b**2+rs**2)).decompose())**-1
    return f

def r_s(M):
    return 1.05/0.65*(M/(1.e8))**0.5

def velocity_kick(w,w_per,impact,mass,rs):
    v_kick = (G*mass*w_per)/(np.sqrt(impact**2 + rs**2) * (w)**2 )
    v_kick=v_kick.decompose()
    return np.abs(v_kick.value * 10**-3)


def calculate_gap_size(model,N,fcrit,Mmin,Mmax,model_mass=None):
    w, wperp =assign_velocities(N)
    high_w_cut =np.where(w<10000)
    n_subhalo=len(high_w_cut[0])
    w=w[high_w_cut]
    wperp=np.abs(wperp[high_w_cut])
    
    # LCDM Case
    if not model_mass:
        M = draw_mass_lcdm_Erkal_2016(Mmin,Mmax,n_subhalo)
        
    # WDM or FDM case 
    else:
        M, _ = sample_mass(model(mass=model_mass),Mmin,Mmax,size=n_subhalo)
    
    t=get_impact_times(n_subhalo)
    rs= r_s(M)
    b = assign_bs(np.max(2*rs),n_subhalo)
    t_c=time_caustic(M*u.Msun,rs*u.kpc, t*u.Gyr, r_stream*u.kpc, w*u.km/u.s, wperp*u.km/u.s, b*u.kpc)
    theta= gap_size(M*u.Msun,rs*u.kpc, t*u.Gyr, r_stream*u.kpc, w*u.km/u.s, wperp*u.km/u.s, b*u.kpc,t_c)
    rho=gap_depth(M*u.Msun, rs*u.kpc, t*u.Gyr, r_stream*u.kpc, w*u.km/u.s, wperp*u.km/u.s, b*u.kpc)
    v_kick=velocity_kick(w*u.km/u.s,wperp*u.km/u.s,b*u.kpc,M*u.Msun,rs*u.kpc)
    
    return theta[(rho < fcrit)*(v_kick > 0.1)]

def get_lcdm_SMHF_integral(Mmin,Mmax):
    mmin=Mmin
    mmax=Mmax
    return np.trapz(y=dN_dM_lcdm_Erkal_2016(mmin,mmax)[0],x=dN_dM_lcdm_Erkal_2016(mmin,mmax)[1])


def plot_gap_size_dist(model,N,fcrit,Mmin,Mmax,mass_arr):
    M=np.linspace(Mmin,Mmax,1000000)
    # LCDM Case 
    lcdm_theta=calculate_gap_size(CDMModel,N,0.9,Mmin,Mmax,model_mass=None)
    all_norm=len(lcdm_theta)
    lcdm_integral=get_lcdm_SMHF_integral(Mmin,Mmax)
    
    fig, ax = plt.subplots(figsize=(12,12))
    hist,bins = np.histogram(lcdm_theta,bins=np.arange(0.,100.,0.25))
    bins = 0.5*(bins[1:]+bins[:-1])
    hist = hist/(np.float(all_norm)*0.25)
    plt.loglog(bins,bins*hist,linewidth=6, linestyle='--',label="LCDM f<0.90")
    

    f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
    for mass_val in mass_arr:
        theta_model=calculate_gap_size(model,N,fcrit,Mmin,Mmax,model_mass=mass_val)
        integral_model=np.trapz(y=model(mass=mass_val).subhalo_mass_function(M)/M,x=M)
        model_norm=lcdm_integral/integral_model
        
        hist_model,bins_model = np.histogram(theta_model,bins=np.arange(0.,100.,0.25))
        bins_model = 0.5*(bins_model[1:]+bins_model[:-1])
        hist_model = hist_model/(all_norm*0.25*model_norm)
        plt.loglog(bins_model,bins_model*hist_model,linewidth=6,label="$FDMm={}eV$".format(f._formatSciNotation('%1.10e' % (mass_val*1e-22))))
    
        print("calculations for" + " " + str(model) +"model_mass=" + " " + str(mass_val) + " " + "done")

    plt.grid()
    plt.xlabel(r'$\Delta \psi_{\rm gap}$ ($^\circ$)',fontsize=18)
    plt.ylabel(r'$dN(<f_{\rm cut})/d \log \Delta \psi_{\rm gap}$',fontsize=18)
    plt.tick_params(labelsize=16)
    plt.legend(loc='best',fontsize=15)
    plt.xlim(1.,100.)
    plt.ylim(1.e-5,1.e0)
    plt.tight_layout()
    plt.show()
########################################################################################################################################################################

########################################## Expected Number of Gaps ######################################################################################################

def expected_gaps(model,N,fcrit,mass_arr):
    gaps=list()
    m0=2.52e7
    a0=3.26e-5
    alpha=-1.9
    M=np.linspace(10**5,10**9,1000000)
    lcdm_integral=get_lcdm_SMHF_integral(10**5,10**9)
    lcdm_theta=calculate_gap_size(CDMModel,100000000,0.9,10**5,10**9,model_mass=None)
    all_norm=len(lcdm_theta)
    
    for mass_val in mass_arr:
        theta_model=calculate_gap_size(model,N,fcrit,10**5,10**9,model_mass=mass_val)
        integral_model=np.trapz(y=model(mass=mass_val).subhalo_mass_function(M)/M,x=M)
        model_norm=lcdm_integral/integral_model
        hist_model,bins_model = np.histogram(theta_model,bins=np.arange(0.,100.,0.25))
        bins_model = 0.5*(bins_model[1:]+bins_model[:-1])
        hist_model = hist_model/(all_norm*0.25*model_norm)
        
        # expected number of gaps 
        
        dlogx = np.diff(np.log(bins_model))
        Y= bins_model*hist_model
        y_mean=np.sqrt(Y[1:]*Y[:-1])
        y_mean=y_mean*dlogx
        expected_gaps=np.trapz(y_mean)
        gaps.append(expected_gaps)
        
        print("calculations for WDM=" + str( mass_val) + "keV" + " " + "done")
    
    return np.array(gaps)