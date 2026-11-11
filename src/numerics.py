"""Continuous uniform mixture: fast screen, directed-decimal certificate.

Only lower bounds from this module may authorize an output. Decimal ln/exp
are correctly rounded; stepping outward encloses them. Basic operations use
directed rounding. No tolerance is added to the acceptance inequality.
"""
from decimal import Decimal as D, Context, ROUND_FLOOR, ROUND_CEILING, ROUND_HALF_EVEN, localcontext
from functools import lru_cache
import math
import numpy as np
from scipy.special import betaln, betainc, betaincc, expit

PREC=42
DOWN=Context(prec=PREC,rounding=ROUND_FLOOR,Emin=-999999999,Emax=999999999)
UP=Context(prec=PREC,rounding=ROUND_CEILING,Emin=-999999999,Emax=999999999)
NEAR=Context(prec=PREC,rounding=ROUND_HALF_EVEN,Emin=-999999999,Emax=999999999)
SCALE=1<<36

class I:
    __slots__=('lo','hi')
    def __init__(self,lo,hi=None):
        self.lo=lo if isinstance(lo,D) else D(str(lo))
        self.hi=self.lo if hi is None else (hi if isinstance(hi,D) else D(str(hi)))
        if self.lo>self.hi: raise ValueError('Reversed interval')
    def __add__(self,o):
        o=iv(o);return I(DOWN.add(self.lo,o.lo),UP.add(self.hi,o.hi))
    __radd__=__add__
    def __neg__(self): return I(self.hi.copy_negate(),self.lo.copy_negate())
    def __sub__(self,o): return self+-iv(o)
    def __rsub__(self,o): return iv(o)+-self
    def __mul__(self,o):
        o=iv(o); pairs=[(a,b) for a in (self.lo,self.hi) for b in (o.lo,o.hi)]
        return I(min(DOWN.multiply(a,b) for a,b in pairs),max(UP.multiply(a,b) for a,b in pairs))
    __rmul__=__mul__
    def __truediv__(self,o):
        o=iv(o)
        if o.lo<=0<=o.hi: raise ZeroDivisionError('interval contains zero')
        rec=I(DOWN.divide(D(1),o.hi),UP.divide(D(1),o.lo))
        return self*rec
    def __rtruediv__(self,o): return iv(o)/self
    def exp(self):
        a=NEAR.exp(self.lo);b=NEAR.exp(self.hi)
        return I(NEAR.next_minus(a),NEAR.next_plus(b))
    def log(self):
        if self.lo<=0: raise ValueError('log nonpositive')
        a=NEAR.ln(self.lo);b=NEAR.ln(self.hi)
        return I(NEAR.next_minus(a),NEAR.next_plus(b))
    def floor_scaled(self): return int(DOWN.multiply(self.lo,D(SCALE)).to_integral_value(rounding=ROUND_FLOOR))
    def ceil_scaled(self): return int(UP.multiply(self.hi,D(SCALE)).to_integral_value(rounding=ROUND_CEILING))

def iv(x): return x if isinstance(x,I) else I(x)

def loglik_i(n,s,v):
    if n==0: return I(0)
    return s*v.log()+(n-s)*(1-v).log()

@lru_cache(maxsize=64)
def rr_i(epsilon):
    a=1/(1+I(epsilon).exp());return a,1-a

def rr(epsilon):
    a=float(expit(-epsilon));return a,1-a

def kl(p,q):
    p=np.asarray(p);q=np.asarray(q)
    return np.where(p>0,p*np.log(np.maximum(p,1e-300)/q),0)+np.where(p<1,(1-p)*np.log(np.maximum(1-p,1e-300)/(1-q)),0)

def fast_stats(N,S,epsilon):
    N=np.asarray(N);S=np.asarray(S);a,b=rr(epsilon)
    raw=np.divide(S,N,out=np.full(len(N),.5),where=N>0)
    phat=np.clip(raw,a,b)
    lh=S*np.log(phat)+(N-S)*np.log1p(-phat)
    # Select the tail difference that avoids cancellation in incomplete beta.
    u=betainc(S+1,N-S+1,b)-betainc(S+1,N-S+1,a)
    v=betaincc(S+1,N-S+1,a)-betaincc(S+1,N-S+1,b)
    mass=np.maximum(u,v)
    with np.errstate(divide='ignore'):
        lq=betaln(S+1,N-S+1)+np.log(mass)-math.log(b-a)
    # Underflowed screens fall back to a broad valid numerical lower bound.
    for e in np.flatnonzero(~np.isfinite(lq)):
        _,_,qi,_,_=verified_edge(int(N[e]),int(S[e]),str(epsilon))
        lq[e]=float(qi.lo)
    lq[N==0]=0.;lh[N==0]=0.
    return phat,lh,lq

@lru_cache(maxsize=100000)
def verified_edge(n,s,epsilon,resolution=8):
    """Return (MLE interval, log Lhat interval, log Q_lower interval,
    rigorous log integral enclosure width, number of integration cells).

    Concavity of log L gives an integral lower bound from the exponential
    of every endpoint chord. The upper rectangle uses the interval maximum.
    Q_lower is the exact decimal lower endpoint, fixed before subsequent logs.
    """
    a,b=rr_i(epsilon)
    if n==0: return I('.5'),I(0),I(0),0.,0
    raw=I(s)/n
    if raw.hi<a.lo: hat=a
    elif raw.lo>b.hi: hat=b
    else: hat=I(max(raw.lo,a.lo),min(raw.hi,b.hi))
    lh=loglik_i(n,s,hat)
    # Interior integration endpoints. Omitted boundary slivers have nonnegative
    # mass and are included in the upper bound. Grid location is just a proposal.
    af=float(a.hi);bf=float(b.lo);hf=float(hat.lo)
    sd=math.sqrt(max(hf*(1-hf),1e-8)/(n+1))
    points={a.hi,b.lo}
    for j in range(-8*resolution,8*resolution+1):
        z=D(str(hf+j*sd/resolution))
        if a.hi<z<b.lo:points.add(z)
    if a.hi<hat.lo<b.lo:points.add(hat.lo)
    points=sorted(points)
    logs=[loglik_i(n,s,I(z))-lh for z in points]
    lower=I(0);upper=I(0)
    for j,(x,y) in enumerate(zip(points,points[1:])):
        f,g=logs[j],logs[j+1];width=I(y)-I(x)
        diff=g-f
        if diff.lo>0 or diff.hi<0:
            area=width*(g.exp()-f.exp())/diff
            lo=max(D(0),area.lo)
        else:
            lo=(width*I(min(f.lo,g.lo)).exp()).lo
        lower=lower+I(lo)
        top=max(f.hi,g.hi)
        if x<=hat.hi and y>=hat.lo: top=max(top,D(0))
        upper=upper+width*I(top).exp()
    # Since L(v)<=Lhat, missing endpoint slivers have relative mass <= width.
    upper=upper+I(UP.subtract(a.hi,a.lo))+I(UP.subtract(b.hi,b.lo))
    relative_lower=(lower/(b-a)).lo
    relative_upper=(upper/(b-a)).hi
    log_lower=I(relative_lower).log()+lh
    log_upper=I(relative_upper).log()+lh
    # Use a scalar log Q lower bound. Its exponential is <= true Q.
    log_q_lower=I(log_lower.lo)
    enclosure=float(UP.subtract(log_upper.hi,log_lower.lo))
    return hat,lh,log_q_lower,enclosure,len(points)-1

def verified_stats(N,S,epsilon):
    return [verified_edge(int(n),int(s),str(epsilon)) for n,s in zip(N,S)]
