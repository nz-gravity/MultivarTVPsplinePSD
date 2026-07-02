(cid:97)/nz-

g
r
a
v
i
t
y

dev

t
p
i
r
c
s

/multivar_pspline_manu

Multivariate Bayesian P-spline estimation of spectral density matrices,
with application to LISA TDI noise

Avi Vajpeyi1, Renate Meyer1, Patricio Maturana-Russel1,2, and Jianan Liu1
1Department of Statistics, The University of Auckland, Auckland, New Zealand and
2Department of Mathematical Sciences, Auckland University of Technology, Auckland, New Zealand
(Dated: June 12, 2026)

We present a Bayesian P-spline method for estimating the frequency-dependent cross-spectral
density matrix of stationary multivariate time series. The inverse spectral matrix is parametrised
through its frequency-varying Cholesky decomposition, which guarantees Hermitian positive defi-
niteness at every frequency. Each real log-diagonal entry and each real and imaginary off-diagonal
entry is given an independent penalised B-spline prior that controls smoothness. Inference uses a
blocked, coarse-grained Whittle likelihood with safe-Bayes η-tempering to stabilise posterior cali-
bration, sampled by the No-U-Turn Sampler from a variational initialisation. On synthetic VAR(2)
benchmarks with known ground truth, the method recovers both diagonal and cross-spectral struc-
ture, attains near-nominal credible-interval coverage, and achieves a relative integrated squared
(Frobenius) error (RISE) that decreases with sample size. We then apply the method to publicly
released simulated LISA time-delay interferometry (TDI) data in two noise configurations. In the
idealised symmetric case, the full multivariate model and a reduced model that assumes a diagonal
aet noise covariance agree to within ∼ 10−3 in RISE. Under realistic noise that is asymmetric across
the six Movable Optical Sub-Assemblies (MOSAs), the aet-diagonal assumption fails by more than
an order of magnitude in RISE (∼ 3.3×10−2 versus ∼ 10−3), whereas the full multivariate model
recovers the cross-spectral structure.

I.

INTRODUCTION

Accurate estimation of the noise power spectral density
(PSD) underpins gravitational-wave (GW) data analysis:
search, parameter estimation, and stochastic-background
inference all require a reliable noise model across the
detector band. For LIGO and Virgo, parametric mod-
els augmented with spline components, most notably
BayesLine [1], are widely used. The space-based detec-
tor LISA [2] poses a harder problem: its three time-delay
interferometry (TDI) channels are correlated, and their
noise properties vary smoothly but non-trivially across
the millihertz band. A single-channel noise model can-
not capture the cross-spectral correlations that matter
for joint inference.

Flexible nonparametric Bayesian approaches to PSD
estimation have a well-developed statistical foundation,
characterised by an expansion of the PSD in terms of
a potentially large or infinite number of basis functions
combined with noninformative priors on the coeﬀicients.
Examples include Bernstein polynomials with weights
induced by Dirichlet process priors [3], cosines with a
discounted regularized horseshoe prior [4], and B-splines
with knots and weights induced by Dirichlet process pri-
ors [5]. With an a priori unknown and potentially in-
finite number of basis functions, posterior computation
requires trans-dimensional Markov chain Monte Carlo
(MCMC) algorithms, which are computationally expen-
sive. Penalised B-splines (P-splines) [6] avoid the trans-
dimensional step by representing smooth functions as
linear combinations of a finite but large number of B-
spline basis functions, with a roughness penalty on adja-
cent coeﬀicients that prevents overfitting. In a Bayesian

formulation this penalty can be incorporated naturally
into the prior distribution of the spline coeﬀicients. Ap-
plied to spectral density estimation via the Whittle likeli-
hood, this yields a smooth, data-driven estimator that re-
quires no parametric noise template. Univariate Bayesian
P-spline PSD methods were introduced by Maturana-
Russel and Meyer [7]. They have since been applied
to GW noise estimation, including recent LISA-specific
work that uses log-spline representations tuned to the
steep low-frequency rise of LISA noise [8]. Related non-
parametric strategies model unknown noise shapes in
stochastic-background searches with other spline fami-
lies rather than penalised B-splines. These include cubic
B-spline single-link noise models analysed by Bayesian
model selection [9], and Akima-spline noise components
introduced at the TDI level with the number and location
of knots inferred by a reversible-jump MCMC [10].

These univariate approaches model each channel in-
dependently. This is justified in the idealised equal-
arm stationary LISA configuration, which has vanishing
off-diagonal cross-spectra between the three TDI chan-
nels, but realistic noise — with unequal arm lengths,
noise-level imbalance across the six Movable Optical Sub-
Assemblies (MOSAs), and instrumental breathing [11] —
generates non-trivial off-diagonal entries in the 3(cid:2)3 spec-
tral density matrix S(f ), whose normalised magnitudes
(the channel-pair coherences) quantify the cross-channel
correlation that single-channel models discard. Treating
these cross-spectra as zero when they are not can de-
grade sensitivity to stochastic GW backgrounds and bias
parameter estimation [12, 13]. Any extension of univari-
ate methods to nonparametrically model S(f ) needs to
ensure Hermitian positive definiteness at every frequency

while allowing each matrix element to vary smoothly with
frequency.

II. MULTIVARIATE P-SPLINES

2

Existing nonparametric treatments of LISA TDI spec-
tral matrices follow two broad routes. Muratore et al.
[14] model the diagonal PSDs and the real and imaginary
parts of the cross-spectra of S(f ) as smooth fractional de-
viations from a design spectrum, each represented by a
natural cubic spline. They note that this construction is
not guaranteed to yield a positive-definite matrix away
from the design point, and that it is well suited to their
local Fisher-matrix study of noise-knowledge uncertainty
rather than to fitting data directly. A second route mod-
els the single-link noise log-PSD nonparametrically with
cubic B-splines and maps it to the TDI variables through
the known TDI transfer functions [9]. Because the re-
sulting TDI covariance is built from a valid single-link
covariance through this linear transfer, it is Hermitian
positive definite by construction, at the cost of assuming
the simplified single-link noise structure.

In this paper, we model the PSD matrix of the xyz
TDI variables directly by extending the Bayesian P-spline
PSD framework [7] to the multivariate p-dimensional set-
ting. We parametrise S(f )−1 through its Cholesky de-
composition [15, 16], which factorises the posterior den-
sity into p independent components that can be sam-
pled in parallel and guarantees the Hermitian positive
definiteness condition. Each block’s log-diagonal spectra
and real and imaginary parts of the complex off-diagonal
cross-spectra are modelled with penalised B-splines un-
der a hierarchical smoothing prior. This prior is up-
dated by a blocked, coarse-grained Whittle likelihood
with safe-Bayes η-tempering [17] to stabilise posterior
calibration. Each chain is initialised from a draw of a
stochastic variational approximation [18], with sampling
performed by the No-U-Turn Sampler [19]. We validate
the method on a three-channel VAR(2) simulation with
known ground truth, and on publicly released simulated
LISA TDI datasets in a symmetric and a physically real-
istic asymmetric per-MOSA noise configuration. In the
asymmetric case an aet-restricted model fails by more
than an order of magnitude in relative integrated squared
(Frobenius) error while the full multivariate model re-
covers the cross-spectral structure. A further benchmark
against the bivariate estimators of Liu et al. [20] on a
VAR(2) problem with closed-form ground truth is re-
ported in Appendix B.

The remainder of the paper is organised as follows.
Section II develops the statistical model and the infer-
ence scheme. Section III presents simulation studies on
a 3D VAR(2) benchmark and on publicly released simu-
lated LISA TDI data. Section IV summarises the results,
discusses limitations and future directions.

A. Likelihood

A glossary of the symbols used throughout this paper
is provided in Table III (Appendix A). Readers may find
it useful to keep this table open while reading the rest
of this section. Let Z = (Z1, . . . , Zn)⊤ 2 Rn×p be a
p-dimensional stationary, mean-zero time series sampled
at intervals ∆t = 1/(2fN y), so that Zt = Z(t∆t) for
t = 1, . . . , n, where fN y is the Nyquist frequency. The
total observation time is T = n∆t and the frequency
resolution is

∆f =

1
n∆t

=

1
T

.

The discrete Fourier transform (DFT) of Z is

d(fk) = ∆t

(cid:18)

Zt exp

(cid:0)2πi

(cid:19)

,

k
n

t

nX

t=1

(1)

(2)

with fk = k/T for k = 1, . . . , N , where N = bn/2c.

P∞

For stationary series with absolutely summable matrix
τ =−∞kΓ(τ )k < 1, the Fourier coef-
autocovariances
ficients d(fk) are asymptotically independent, complex
Gaussian with mean zero and covariance T S(fk), where

S(f ) =

1
2fN y

∞X

τ =−∞

Γ(τ ) exp ((cid:0)2πif τ ∆t)

is the two-sided spectral density matrix — the Fourier
transform of the autocovariance Γ(τ ) = E(ZtZ⊤
t+τ ) at
integer lag τ . While the likelihood below is expressed in
terms of the two-sided density, all spectra reported in this
paper (estimates, analytic references, and figures) follow
the one-sided convention, obtained by doubling the two-
sided density at positive frequencies. This asymptotic
Gaussian approximation is the basis of the multivariate
Whittle likelihood [21]

L(djS) /

NY

jS(fk)j−1 (cid:2)

k=1
(cid:18)

exp

(cid:0) 1
T

(cid:19)

d(fk)∗S(fk)−1d(fk)

,

(3)

where d(fk)∗ denotes the conjugate transpose of d(fk)
and j(cid:1)j denotes the matrix determinant. In general a spec-
tral density matrix is only guaranteed to be Hermitian
positive semidefinite. The Whittle likelihood, however,
requires S(fk)−1 to exist, so we assume strict positive def-
initeness throughout (a property the Cholesky parametri-
sation of Section II B enforces by construction).

The diagonal entries Sii(f ) are the power spectral den-
sities (PSDs) of channel i, while the off-diagonals Sij(f )
for i 6= j are the (complex) cross-spectra. A scale-free

summary of the linear dependence between channels i
and j at frequency f is the complex coherence,

Nb assumed-independent single-block periodogram con-
tributions:

Cij(f ) =

p

Sij(f )
Sii(f ) Sjj(f )

,

jCij(f )j 2 [0, 1], (cid:97) (4)

I(i)(fk) (cid:24) CW p(TbS(fk), 1),
Y(fk) (cid:24) CW p(TbS(fk), Nb).

(9)
(10)

3

whose magnitude is zero when channels i and j are uncor-
related at frequency f and unity when they are perfectly
linearly related. We use jCij(f )j to visualise off-diagonal
recovery in the LISA application (Sec. III B).

Blocking. To trade frequency resolution for peri-
odogram stability we partition Z into Nb equal, non-
overlapping blocks (Z(1), . . . , Z(Nb)), each of duration
Tb = T /Nb and containing n/Nb samples (illustrated in
Fig. 1). Each block may be tapered by a window function
wt before the DFT to suppress spectral leakage. Writing
the number of samples per block as nb = n/Nb, we define
the window’s equivalent noise bandwidth as

ENBW = nb

P

(

P

nb
t=1 w2
t
t=1 wt)2 . (cid:97)
nb

(5)

We rescale each block periodogram by 1/ENBW, so that
the complex-Wishart mean of Y(fk) remains Tb S(fk).
The separate effect of tapering on the effective likelihood
information is described below. Denoting the DFT of
block i by d(i), and assuming blocks are approximately
independent under stationarity, the blocked Whittle like-
lihood factorises as

Lb(djS) =

NbY

i=1

L(d(i)jS) .

(6)

Collecting the sum over blocks into a single Hermitian
matrix

Y(fk) =

NbX

i=1

I(i)(fk) = Nb ¯I(fk),

(7)

where I(i)(fk) = d(i)(fk)d(i)(fk)∗ is the block-i peri-
odogram and ¯I(f ) = N −1
Nb
i=1 I(i)(f ) is the block-
averaged periodogram (the Welch estimator), and apply-
ing the trace cyclic property tr(ABC) = tr(CAB), the
blocked likelihood reduces to

P

b

Lb(djS) /

n/(2Nb)Y

jS(fk)j−Nb (cid:2)

k=1
(cid:16)

(cid:0) 1
Tb

(cid:2)
tr

exp

S(fk)−1Y(fk)

(cid:3)(cid:17)

.

(8)

Under the asymptotic Gaussian approximation, Y(fk)
follows a complex Wishart distribution1 from summing

1 We write A ∼ CW d(M, m) if the d×d Hermitian positive definite
matrix A has density proportional to |M|−m exp(−tr[M−1A]),
with mean mM.

p

P

Eigendecomposition. Because Y(fk)

is Hermitian
it admits an eigendecomposition
positive semidefinite,
ν v(k)∗
ν=1 u(k)
ν v(k)
ν=1 λ(k)
, with
Y(fk) =
rescaled eigenvectors u(k)
ν . Substituting
into the trace in (8) gives the eigenvector form of the
blocked likelihood,

ν u(k)∗

ν v(k)
λ(k)

ν =

=
q

P

p

ν

ν

Lb(djS) /

n/(2Nb)Y

k=1

jS(fk)j−Nb

(cid:2) exp

(cid:0) 1
Tb

pX

ν=1

!

ν S(fk)−1u(k)
u(k)∗

ν

.

(11)

This follows from the spectral decomposition of the Her-
mitian positive semidefinite Y(fk) and the linearity of
the trace. The eigenvector representation is used com-
putationally because it turns the trace contribution into
a sum of quadratic forms in the Cholesky parameters,
which is the form that factorises into the per-channel re-
gressions below.

f ∈Jh

h =

Coarse graining. Blocking reduces the number of pe-
riodogram matrices from N to n/(2Nb), but each of
the remaining frequencies still requires a separate eval-
uation of S(fk)−1. Where S(f ) is slowly varying, we
can pool adjacent frequencies to gain further speed (il-
lustrated in Fig. 1). Partition the block frequencies
Nc
into Nc disjoint subsets ff1, . . . , fn/(2Nb)g =
h=1 Jh,
each containing Nh adjacent frequencies with bin centre
¯fh (cid:17) 1
f , which need not coincide with a Fourier
Nh
frequency. Replace the per-frequency matrix Y(fk) by
P
the within-bin sum Ycg

Y(f ).

f ∈Jh

P

S

Under the assumption that S(f ) (cid:25) S( ¯fh) across each
Jh, the Y(f ) within the bin are approximately i.i.d.
CW p(TbS( ¯fh), Nb), and by the additivity of the complex
Wishart under a common scale matrix,

(cid:0)

Ycg

h ˙(cid:24) CW p

Tb S( ¯fh), NbNh

(cid:1)

. (cid:97)

(12)

The approximation is only as good as the constant-S as-
sumption, so bin edges must avoid sharp spectral features
(peaks, lines, transfer-function nulls).

The coarse-grained likelihood then takes the same form
as Eq. (11) with the n/(2Nb) block frequencies replaced
by Nc coarse bins and a Wishart shape parameter NbNh
(the number of fine periodograms summed per coarse
bin):

(cid:12)
(cid:12)−NbNh (cid:2)

Lb,c(djS) /

NcY

h=1
(cid:16)

(cid:12)
(cid:12)S( ¯fh)
(cid:2)
tr

exp

(cid:0) 1
Tb

S( ¯fh)−1Ycg
h

(cid:3)(cid:17)

, (cid:97)

(13)

4

FIG. 1. Construction of the blocked, coarse-grained periodogram. The n-sample time series is partitioned into Nb non-
overlapping blocks {bi} of length T /Nb. Each block is Fourier-transformed independently, yielding Nb single-block periodograms
Nb
i=1 I(i)(fk) (related
I(i) each of length n/(2Nb). These are summed across blocks to form the blocked periodogram Y(fk) =
to the Welch estimator by ¯I = Y/Nb), which is then summed over consecutive groups of Nh adjacent fine frequencies to produce
the final coarse-grained periodogram Ycg

∑

h of length n/(2NbNh).

with eigendecomposition

Ycg

h =

pX

ν=1
q

ν u(h)∗
u(h)

ν

,

u(h)

ν =

λ(h)
ν v(h)
ν

.

(14)

(15)

Although rescaling the periodogram by 1/ENBW pre-
serves its mean, tapering correlates neighbouring Fourier
coeﬀicients and therefore reduces the effective number
of independent frequency-domain observations. We ap-
proximate this loss of information by dividing the log-
likelihood by the window’s equivalent noise bandwidth
in Eq. (5), or equivalently by using the power likelihood

Lb,c,w(d j S) / Lb,c(d j S)1/ENBW . (cid:97)

(16)

Thus a rectangular window leaves the likelihood un-
changed, whereas a non-rectangular taper reduces its cur-
vature by the factor 1/ENBW.

B. Parametrisation using the Cholesky
Decomposition

Following Rosen and Stoffer [15], Hu and Prado
[16], we parametrise the inverse spectral density matrix
through its Cholesky decomposition. Working with S−1
rather than S is what makes the problem tractable: the

quadratic form u∗S−1u splits into a sum of p decoupled
quadratic terms, which in turn factorises the likelihood
into p independent regressions (one per channel) that can
be sampled in parallel. We write

where Dk
δ2
1k, δ2

2k, . . . , δ2

is
pk, and
0

S(fk)−1 = T∗

k D−1
diagonal with

k Tk,

positive

B
B
B
B
B
B
@

Tk =

1
(cid:0)θ(k)
21
(cid:0)θ(k)
31
...
(cid:0)θ(k)
p1

0
1
(cid:0)θ(k)
32
...
(cid:0)θ(k)
p2

0
0

1
(cid:1) (cid:1) (cid:1)
0
C
(cid:1) (cid:1) (cid:1)
0
C
C
...
. . .
C
1
C
C
. . .
. . .
A
0
(cid:1) (cid:1) (cid:1) (cid:0)θ(k)
p,p−1 1

(17)
entries

(18)

is a p (cid:2) p complex unit lower triangular matrix with θ(k)
il
representing the complex-valued off-diagonal elements for
i > l.

This parametrisation factorises the likelihood into a

product of p per-channel terms,

L(djS) /

pY

j=1

Lj(θj, δj),

(19)

one for each row of T. Each Lj is a univariate Whittle
regression of the jth channel of the eigenvector pseudo-
observations onto the preceding (j (cid:0) 1) channels:

Lj(θj, δj) /

0

B
@

NcY

h=1

δ−2NbNh
jh

exp

P

(cid:0)

p
ν=1

(cid:12)
(cid:12)
(cid:12)u(h)
jν

P

(cid:0)

j−1
l=1 θ(h)

jl u(h)

lν

Tb δ2
jh

2

(cid:12)
(cid:12)
(cid:12)

1

C
A , (cid:97)

(20)

ZTimeseriesnsamplest=0t=Tb1b2b3...bNbBlockspartitionintoNbI(1)I(2)I(3)...I(Nb)Blockedperiodogramsn/(2Nb)freqs/blockFFTsumoverblocksY(fk)=∑Nbi=1I(i)(fk)BlockedperiodogramY(fk)Coarse-grainedperiodogramYcghNc=n/(2NbNh)binssumeveryNhfrequenciesYcgh=∑fk∈JhY(fk)with regression coeﬀicients
off-diagonal
Cholesky entries at coarse bin h) and noise variance δ2
jh
(the jth diagonal entry of Dh). The eigenvectors u(h)
are those of the coarse-grained statistic Ycg
h , evaluated

(the

ν

θ(h)
jl

5

at the bin midpoint ¯fh. The parameter blocks θj and
δj collect all θ(h)
(l < j) and δjh across the coarse bins
jl
h = 1, . . . , Nc.

In particular, for p = 3 channels (as in the LISA case),
the three factors depend on the parameter subvectors δ1,
(δ2, θ2), and (δ3, θ3), respectively, as follows:

L1(δ1) /

L2(θ2, δ2) /

L3(θ3, δ3) /

0

B
@

0

B
@

0

B
@

NcY

h=1

NcY

h=1

NcY

h=1

δ−2NbNh
1h

exp

δ−2NbNh
2h

exp

δ−2NbNh
3h

exp

1

C
A ,

P

P

(cid:0)

(cid:0)

2

(cid:12)
(cid:12)
(cid:12)

(cid:12)
(cid:12)
(cid:12)u(h)
1ν

p
ν=1
Tb δ2
1h
(cid:12)
(cid:12)
(cid:12)u(h)
2ν
Tb δ2
2h

p
ν=1

(cid:0) θ(h)

21 u(h)

1ν

P

(cid:0)

p
ν=1

(cid:12)
(cid:12)
(cid:12)u(h)
3ν

31 u(h)

1ν

(cid:0) θ(h)
Tb δ2
3h

2

(cid:12)
(cid:12)
(cid:12)

1

C
A ,

(cid:0) θ(h)

32 u(h)

2ν

(21)

(22)

(23)

2

(cid:12)
(cid:12)
(cid:12)

1

C
A .

Note that a factorised likelihood does not in general im-
ply a factorised posterior. Here, however, the posterior
also factorises because we place independent priors across
these three parameter blocks,

π(δ1, δ2, θ2,δ3, θ3) =

π1(δ1) π2(δ2, θ2) π3(δ3, θ3),

(24)

p(δ1, δ2, θ2,δ3, θ3 j d) /

L1(δ1) π1(δ1)(cid:2)
L2(δ2, θ2) π2(δ2, θ2)(cid:2)
L3(δ3, θ3) π3(δ3, θ3).

(25)

Hence Bayes’ rule yields three independent posterior fac-
tors corresponding to δ1, (δ2, θ2), and (δ3, θ3). This
means we can run an MCMC chain for each block sepa-
rately and in parallel, which in general reduces the com-
putation time.

C. Safe-Bayes η-tempering

The Whittle likelihood is an asymptotic approxima-
tion to the exact Gaussian likelihood of Z, and the
coarse-grained form in Eq. (13) compounds this with a
within-bin constant-S assumption. Both approximations
introduce mild misspecification. A well-known conse-
quence of likelihood misspecification is that the posterior
can over-concentrate, producing credible intervals that
under-cover the truth even when the posterior mode is
accurate [17].

We mitigate this with the safe-Bayes correction of
Grünwald and van Ommen [17]: raise the likelihood to a

power η 2 (0, 1],

Lη(d j S) / Lb,c,w(d j S)η / Lb,c(d j S)η/ENBW, (cid:97)

(26)
which, together with the ENBW correction, gives an ef-
fective scaling NbNh 7! ηNbNh/ENBW in Eq. (13). The
limit η ! 1 recovers the window-corrected likelihood,
while η ! 0 recovers the prior.
Intermediate η down-
weights the (asymptotic) likelihood relative to the prior
in exactly the right amount to restore nominal coverage
when the approximation is imperfect.

A useful way to read this correction is through the
curvature of the log-likelihood. From Eq. (13), the ob-
served Fisher information per coarse bin scales linearly
with the number of fine periodograms summed in each
bin: schematically,

(cid:0) ∂2

θθ log Lb,c,w(d j S(θ)) / NbNh
ENBW

I0(θ),

(27)

where I0 is the per-bin information matrix induced by
a single complex Wishart observation. For a fixed win-
dow, increasing either blocking or coarse-graining there-
fore sharpens the log-posterior by the same factor NbNh,
while tapering offsets this by 1/ENBW. We treat η as
a tunable calibration knob rather than a fundamental
quantity. For the small synthetic benchmarks in Sec-
tion III A (n = 16,384, NbNh ≲ 102) the un-tempered
limit η = 1 is already well-calibrated and is used through-
out. The correction becomes important for the LISA ap-
plication (Section III B), where NbNh is two to three or-
ders of magnitude larger and the resulting log-likelihood
curvature drives both posterior over-concentration and,
at suﬀiciently large NbNh, sampler pathologies (maxi-
mum tree-depth saturation in NUTS). An η-sensitivity

study reported in Appendix C motivates the choice of η
used for the LISA results.

D. P-spline modelling

We model the frequency-dependent Cholesky parame-

ters with penalised B-splines,

KjX

m=1
KjlX

m=1
KjlX

log δ2

jk =

<[θjl(fk)] =

=[θjl(fk)] =

Bm(fk)w(δ)
j,m

Bm(fk)w(ℜ)

jl,m

Bm(fk)w(ℑ)

jl,m (cid:97)

(28)

(29)

(30)

m=1
where Bm(fk) are B-spline basis functions evaluated at
frequency fk, Kj and Kjl are the numbers of basis func-
tions for the j-th diagonal and (j, l) off-diagonal compo-
nents, respectively, and w(δ)
jl,m are the corre-
sponding spline coeﬀicients. When the blocked, coarse-
grained likelihood is used, these expansions are evaluated
at the coarse-bin centres ¯fh, yielding the parameters δ2
jh
and θ(h)
jl

that enter Eq. (20).

jl,m, w(ℑ)

j,m, w(ℜ)

Knots are placed adaptively, in the spirit of Maturana-
Russel and Meyer [7]: for each Cholesky component the
empirical periodogram is denoised, the absolute gradi-
ent of the result — plus a small uniform floor so that
featureless regions still receive knots — is normalised to
a probability mass function over frequency, and knots
are positioned at equally spaced quantiles of its cumula-
tive distribution. (cid:97) For the off-diagonal entries the real
and imaginary parts are treated as separate components,
with knots placed independently on j<[θjl(fk)]j and
j=[θjl(fk)]j. Knots therefore cluster where the Cholesky
component varies rapidly and spread out in smooth re-
gions, with no manual tuning.

Each component j has its own penalty matrix Pj with

entries

Z

1

[Pj]lm =

B′′

l (t) B′′

m(t) dt. (cid:97)

(31)

0
Unlike the finite-difference approximation D⊤
2 D2 used
in the original P-spline formulation of Eilers and Marx
[6], this integral form is exact for non-uniform knot se-
quences [22]. Note that the integral form of Pj is the
same for every component. The j-subscript reflects only
that the knot sequence and basis size Kj are placed adap-
tively per component, so the resulting Kj (cid:2) Kj matrix
differs across j.

6

spline coeﬀicients of every Cholesky component. Second,
we combine these priors with the η-tempered, blocked,
coarse-grained Whittle likelihood (Eq. 26) using the per-
channel factorisation of Eq. (19). Third, we warm-start
each MCMC chain from a draw of a stochastic vari-
ational approximation to the posterior, to bypass the
expensive NUTS adaptation in low-mass-curvature re-
gions. Fourth, we sample the warm-started posterior
with NUTS and assess convergence with standard chain
diagnostics. Each stage is described below.

We place hierarchical priors on the spline coeﬀicients
to enforce smoothness. Each spline component (the di-
agonal log-variances log δ2
j and the real and imaginary
parts of each off-diagonal entry θjl) is assigned its own
pair of precision hyperparameters; with a slight abuse of
notation we index a generic such component by j:

ePj)−1)

wjjϕj (cid:24) N (0, (ϕj
ϕjjνj (cid:24) Gamma(αϕ, νjβϕ)
νj (cid:24) Gamma(αν, βν) (cid:97)

(32)
(33)
(34)

where wj is the vector of spline coeﬀicients and ϕj, νj are
precision hyperparameters.2 We use ePj = Pj + ϵ I with
ϵ = 10−6 to ensure the prior precision is strictly positive
definite. Unless stated otherwise, the results below use
αϕ = βϕ = 1 and αν = βν = 1.

Combining the η-tempered likelihood (Eq. 26) with
these priors and the Cholesky factorisation of S−1,
the posterior is sampled using the No-U-Turn Sam-
pler [NUTS; 19]. The Cholesky parametrisation guar-
antees Hermitian positive definiteness of bS(f ) at every
frequency by construction.

a. Computational complexity. The B-spline basis
matrices are evaluated once at the start of a run, af-
ter the knot locations are fixed, and reused for ev-
ery gradient evaluation. Each call to log Lη then costs
O(p2 Nc K) floating-point operations (using a common
basis size K = Kj = Kjl
for all components): Nc
coarse bins, K basis functions per component, and p2
spline components — the p diagonal log-variances log δ2
jh
plus the real and imaginary parts of the p(p (cid:0) 1)/2 off-
diagonal Cholesky entries θjl. Memory is dominated
by the same p2 Nc K basis-matrix storage.
In practice
this per-evaluation cost is small compared to the fixed
overheads of NUTS warm-up and SVI initialisation, so
wall-clock runtime is dominated not by the gradient-
evaluation cost but by the number of leapfrog integrator
steps required to traverse the posterior. As a rough scale,
a single evaluation of log Lη on the 1-year LISA configu-
ration (p = 3, Nc = 1024, K = 100) takes a fraction of a
millisecond on a single CPU core. Empirical end-to-end

E. Bayesian Inference

The full inference workflow proceeds in four stages.
First, we place hierarchical smoothing priors on the

2 In the closely related univariate constructions of Maturana-
Russel and Meyer [7] and Aimen et al. [8] this hyperparameter is
denoted δj . We use νj here to avoid collision with the Cholesky
diagonal entries δ2

jk introduced in Section II above.

wall-clock numbers are reported alongside the tables in
Section III.

with

b.

Implementation. We implement the model

in
JAX [23] with NumPyro [24], exploiting automatic
differentiation and just-in-time (JIT) compilation for
gradient-based sampling. Throughout we use degree-2
B-splines with second-order derivative penalties (the in-
tegral form of Eq. (31)). Positive precision parameters
ϕj and νj are sampled on the log scale to keep the sam-
pler away from the boundary at zero, and the frequency
grid is standardised to [0, 1] for numerical stability of
the B-spline basis evaluation. To initialise the spline
parameters we first run stochastic variational inference
[SVI; 18] with a low-rank Gaussian guide (rank 16), op-
timised with Adam [25] at learning rate 10−3 for 2000
iterations. (cid:97) Four NUTS chains are then initialised at in-
dependent draws from the resulting variational posterior.
The NUTS mass matrix is adapted during 2000 warm-
up steps, followed by an additional 2000 sampling steps
per chain. These sampler settings are used for both the
simulation study (Section III A) and the LISA applica-
tion (Section III B). Only the basis size K, the number
of blocks Nb, and the tempering η vary between the two.
c. Convergence diagnostics. We assess convergence
by checking that the chains satisfy ˆR < 1.01 (the rank-
normalised Gelman–Rubin potential scale reduction fac-
tor [26, 27]), that the bulk and tail effective sample sizes
both exceed 400, that the energy-Bayesian fraction of
missing information satisfies E-BFMI > 0.3, and that di-
vergent transitions and maximum tree-depth saturation
are absent or negligible. All results in Section III satisfy
these criteria unless explicitly noted otherwise.

III. APPLICATION

A. Simulation study

We summarise spectral estimation accuracy with two
complementary metrics. The first is the relative inte-
grated squared (Frobenius) error (RISE),
R (cid:13)
(cid:13)
(cid:13) bS(f ) (cid:0) S(f )
kS(f )k2

RISE =

(35)

, (cid:97)

(cid:13)
(cid:13)
(cid:13)

df

R

2

F
F df

where k(cid:1)kF denotes the Frobenius norm and the integrals
are evaluated numerically using Simpson’s rule over the
periodogram frequency grid. This metric measures the
bias of the posterior-median estimator bS(f ) relative to
the true spectrum S(f ). The second metric is the em-
pirical coverage at level α, defined as the fraction of (fre-
quency, matrix-element) pairs for which the true values
lie inside the pointwise α credible band, averaged over
realisations. This metric measures the calibration of the
credible intervals. We use α = 0.9 throughout. (cid:97)

We simulate from a three-dimensional VAR(2) process,

Xt = A1Xt−1 + A2Xt−2 + εt,

εt (cid:24) N (0, Σ),

(36)

7

(37)

(38)

(39)

2

A1 = diag(0.4, 0.3, 0.2),
(cid:0)0.2 0.5
0.0
0.4 (cid:0)0.1 0.0
0.0 (cid:0)0.1
0.0
3

A2 =

4

2

3

5 ,

4

Σ =

0.25 0.00 0.08
0.00 0.25 0.08
0.08 0.08 0.25

5 .

These coeﬀicient matrices define a stationary process: the
companion matrix has spectral radius 0.774 < 1.

The one-sided theoretical spectral density matrix is

S(f ) =

2
fs

H(f ) =

I (cid:0)

2X

ℓ=1

H(f ) Σ H(f )∗, with

(40)

!−1

Aℓ e−i2πf ℓ/fs

. (cid:97)

(41)

We generate 500 independent realisations of length
n = 16,384 (with a 512-point burn-in) at sampling fre-
quency fs = 1. Each realisation is fit with K = 10 basis
functions for each Cholesky component, while sweeping
Nb 2 f1, 2, 4, 8, 16g and Nh 2 foff, 2, 4, 8, 16, 32g. Safe-
Bayes tempering (Section II C) is left at η = 1 through-
out, with Nh = off denoting no coarse-graining. A rect-
angular window (ENBW = 1) is applied to each block
before computing the DFT. All other settings follow Sec-
tion II E.

Figure 2 shows the full 3 (cid:2) 3 spectral matrix recov-
ered for a representative n = 16,384 realisation, with the
NUTS and SVI posteriors overlaid. Both samplers re-
cover the diagonal PSDs and off-diagonal cross-spectra
with substantially smoother posterior medians than the
raw periodogram, and the two posterior medians track
each other closely (RISE = 0.114 for NUTS versus 0.102
for SVI). The SVI credible bands, however, are narrower
than those of NUTS: for this realisation the empirical
90% coverage drops from 88% under NUTS to 83% under
SVI. This is the well-known under-dispersion of mean-
field/low-rank Gaussian variational approximations, and
it is why we use SVI only as a fast NUTS initialiser and
retain NUTS as the production sampler when calibrated
credible bands are required. That said, the SVI fit it-
self completes in a few seconds and recovers an accurate
posterior median, which is attractive when only a point
estimate of the spectrum is needed.

Table I reports a representative slice of the (Nb, Nh)
grid. Coverage is near-nominal (89.8–90.8%) across every
configuration, and the mean RISE varies by less than 3%
across the grid. The number of retained frequency bins
Nc falls by a factor of up to 32 between the finest (Nb =
1, no coarse-graining) and coarsest (Nb = 16, Nh = 2)
configurations, with no measurable loss of accuracy or
calibration. MCMC convergence is satisfactory across
all cells: ˆR (cid:20) 1.01 for all parameters, negligible divergent
transitions, and E-BFMI > 0.6.

8

travel times. The six inter-spacecraft interferometer (ISI)
and test-mass (TM) link measurements were propagated
through pytdi [32] to produce the TDI 2.0 Michelson
combinations used here.
The noise-orthogonal

(A, E, T ) TDI combinations
are obtained from xyz by the orthonormal rotation
(A, E, T )⊤ = Maet (X, Y, Z)⊤ with

0

@

Maet =

p
(cid:0)1/
p
1/
p
1/

0
2
p
6 (cid:0)2/
p
1/
3

1/
6 1/
3 1/

1

A ,

p
p
p

2
6
3

(42)

which satisfies M ⊤
aetMaet = I and has unit determinant,
so the rotation introduces no log-likelihood Jacobian cor-
rection [33]. Under the idealised equal-arm, equal-noise
assumption the aet basis diagonalises the noise covari-
ance, motivating the diagonal-aet restrictions tested be-
low.

We analyse two noise configurations:

• noise4a [28] — symmetric realisation:

all six
MOSAs are assigned identical Optical Metrology
System (OMS; ISI carrier) and test-mass accel-
eration amplitude spectral densities equal to the
LDC baseline values (2.4 (cid:2) 10−15 m s−2/
Hz and
7.9 (cid:2) 10−12 m/
Hz, respectively). In this regime
the 3(cid:2)3 cross-spectral matrix is close to degenerate:
SXX (cid:25) SY Y (cid:25) SZZ, with off-diagonals equal in mag-
nitude. A closed-form reference is available from
the analytic TDI transfer of the link-level ASDs.

p

p

FIG. 2. Representative 3 × 3 spectral density matrix esti-
mate for a single n = 16,384 VAR(2) realisation, fit with
K = 10, Nb = 4, Nh = 4 and η = 1 (NUTS: RISE = 0.114,
90% CI coverage 88%; SVI: RISE = 0.102, 90% CI cover-
age 83%). Each panel shows the coarse-grained periodogram
statistic Ycg
h /(TbNbNh) (grey), the theoretical spectrum (dot-
ted black), and posterior medians with 90% credible bands for
NUTS (blue) and SVI (orange). Diagonal panels (S11, S22,
S33) show the power spectral densities, while the off-diagonal
panels show the real and imaginary parts of the cross-spectra
on a linear vertical scale (the VAR(2) PSDs span less than
one decade, so all panels share a linear scale).

(cid:97)

Wall-clock runtime is essentially constant across the
entire grid (8.7–9.6 s on a 4-core CPU node) despite the
32-fold variation in Nc. At this problem size (K = 10,
p = 3, n = 16,384) the per-gradient cost is small enough
that runtime is dominated by fixed NUTS warm-up, JIT
compilation, and SVI initialisation overheads rather than
by the O(p2 Nc K) likelihood evaluation. The benefit of
coarse-graining at this scale is therefore primarily statis-
tical (a smaller effective NbNh makes the posterior easier
to sample) rather than computational. Coarse-graining
begins to pay off in wall-clock terms once Nc or K is
large enough that the per-gradient cost exceeds the fixed
overheads, as in the LISA application below.

B. LISA

We apply the multivariate P-spline model to the
three second-generation time-delay interferometry (TDI)
Michelson channels X2, Y2, Z2 from a publicly released
LISA TDI dataset [28, 29]. All channels are TDI 2.0
throughout, and we suppress the generation subscript in
what follows, writing xyz for (X2, Y2, Z2). The dataset
was produced with lisainstrument [30] under the LISA
[31] at a
Data Challenge (LDC) Spritz noise model
sampling rate of fs = 0.5 Hz and constant 8.3 s light

• noise5a [29] — asymmetric realisation:

the six
per-MOSA OMS and test-mass ASDs are drawn
independently from U (0.5, 2.0)(cid:2)baseline with a
fixed seed.
The resulting cross-spectral ma-
trix has distinct diagonals and non-trivial off-
diagonal structure, representing the physically rele-
vant regime where mis-modelled cross-spectral cor-
relations would degrade sensitivity to stochastic
backgrounds. A closed-form 3 (cid:2) 3 reference is
built with the SEGWO package [34] by feeding the
twelve measured per-MOSA ASDs through ana-
lytic OMS/TM filter shapes and projecting through
PyTDI’s TDI 2.0 Michelson xyz combinations to
obtain the full 3 (cid:2) 3 xyz covariance at every fre-
quency.

We fit each dataset under three competing hypotheses for
the 3 (cid:2) 3 noise cross-spectrum, designed to test whether
the xyz !aet rotation diagonalises the LISA noise co-
variance to within statistical resolution. Since Maet is
orthogonal with unit determinant, fitting in the aet ba-
sis is equivalent to fitting in the xyz basis up to a deter-
ministic relabelling of S. Wherever the aet basis is used
(H0 and H1), the xyz !aet rotation is applied to the
time series first, and all subsequent blocking, averaging,
and coarse-graining is performed in the fitting basis. The
three hypotheses therefore form a strictly nested hierar-
chy over the families of representable spectral matrices,

0.20.512PSD [1/Hz]S11202{S12}0.250.000.250.500.751.00{S13}0.40.20.00.2Cross-spectrum{S21}0.20.512S220.20.00.20.40.60.8{S23}0.20.4Frequency [Hz]0.500.250.000.250.500.75Cross-spectrum{S31}0.20.4Frequency [Hz]0.40.20.00.20.40.6{S32}0.20.4Frequency [Hz]0.20.51S33PeriodogramNUTS 90% CIVI 90% CITruthTABLE I. 3D VAR(2) simulation results at n = 16,384, K = 10, degree-2 splines (mean ± std over 500 realisations), all runs
at η = 1. Coverage is the fraction of (frequency, matrix-element) pairs inside the pointwise 90% credible interval. RISE is
the relative mean squared (Frobenius) error. ESS is the median effective sample size across parameters. Nh =off denotes no
coarse-graining (Nh = 1).

9

Nb

Nh

1
1
1
1
1
1
2
2
2
2
2
4
4
4
4
8
8
8
16
16

off
2
4
8
16
32
off
2
4
8
16
off
2
4
8
off
2
4
off
2

Nc

8192
4096
2048
1024
512
256
4096
2048
1024
512
256
2048
1024
512
256
1024
512
256
512
256

Coverage
0.900 ± 0.031
0.898 ± 0.031
0.900 ± 0.030
0.904 ± 0.029
0.908 ± 0.029
0.907 ± 0.030
0.898 ± 0.031
0.901 ± 0.030
0.904 ± 0.030
0.907 ± 0.030
0.907 ± 0.031
0.901 ± 0.030
0.904 ± 0.029
0.907 ± 0.030
0.907 ± 0.030
0.904 ± 0.030
0.907 ± 0.030
0.907 ± 0.031
0.905 ± 0.029
0.907 ± 0.030

RISE
0.111 ± 0.004
0.112 ± 0.005
0.112 ± 0.005
0.111 ± 0.005
0.110 ± 0.004
0.109 ± 0.004
0.112 ± 0.005
0.112 ± 0.005
0.111 ± 0.005
0.110 ± 0.004
0.109 ± 0.004
0.112 ± 0.004
0.111 ± 0.005
0.110 ± 0.004
0.109 ± 0.004
0.112 ± 0.005
0.111 ± 0.005
0.110 ± 0.004
0.111 ± 0.004
0.110 ± 0.004

ESS
21, 019 ± 2726
24, 057 ± 2807
26, 352 ± 3081
26, 640 ± 3303
25, 480 ± 3525
24, 011 ± 3500
23, 938 ± 2939
26, 279 ± 2979
26, 405 ± 3361
25, 726 ± 3461
23, 997 ± 3787
26, 053 ± 2906
26, 695 ± 3221
25, 753 ± 3636
24, 036 ± 3622
26, 336 ± 3190
25, 587 ± 3658
24, 000 ± 3787
25, 504 ± 3431
23, 862 ± 3724

Runtime (s)
9.61 ± 1.76
9.21 ± 1.69
9.03 ± 1.56
8.92 ± 1.65
8.81 ± 1.62
8.82 ± 1.70
9.11 ± 1.63
9.03 ± 1.72
8.89 ± 1.67
8.76 ± 1.75
8.81 ± 1.69
9.05 ± 1.71
8.84 ± 1.64
8.75 ± 1.72
8.78 ± 1.68
8.89 ± 1.65
8.78 ± 1.70
8.79 ± 1.70
8.74 ± 1.55
8.84 ± 1.68

H0 (cid:26) H1 (cid:26) H2. The three models also differ in their data
object: H0 uses the three diagonal aet periodograms,
whereas H1 and H2 use the full 3 (cid:2) 3 periodogram ma-
trix (in the aet and xyz bases respectively):

• H0 (diagonal aet). The cross-spectral matrix is
constrained to be exactly diagonal in the aet ba-
sis. The xyz time series is first rotated into aet via
Maet (Eq. 42), and each of the three aet channels
is then fit independently with a univariate (p = 1)
log-P-spline model on the diagonals log δ2
j (j =
1, 2, 3, corresponding to log SAA, log SEE, log ST T )
with K = 100 basis functions each. Only the
log-diagonal expansion Eq. (28) is used. The off-
diagonal expansions Eqs. (29) and (30) are not in-
voked. The data for this model are the three uni-
variate averaged periodograms — the diagonal en-
tries SAA, SEE, ST T of Ycg
h — so the likelihood is
a product of three independent Gamma (univariate
Whittle) likelihoods, one per aet channel; the off-
diagonal periodogram entries never enter the fit.
This is precisely the special case of the complex
Wishart likelihood with a diagonal scale matrix S,
which factorises into the three diagonal Gamma
terms. The diagonal aet posterior is assembled
per-sample as diag(SAA, SEE, ST T ) and rotated
back to the xyz basis via Sxyz = M ⊤
aetSaetMaet
for diagnostics. The xyz off-diagonals carried by
H0 therefore reflect only the deterministic TDI ro-
tation, with no fitted physical cross-power.

• H1 (multivariate aet with restricted off-diagonals).
The cross-spectral matrix is approximately diago-

nal in the aet basis. Unlike H0, the data here
are the full 3 (cid:2) 3 averaged aet periodogram ma-
trix and the likelihood is the complete multivari-
ate complex Wishart — the off-diagonal entries do
enter the fit. What is restricted is the off-diagonal
model, not the data. The full multivariate Cholesky
model (Eqs. 28–30) is fit to the aet-rotated time
series with K = 100 basis functions on the diag-
onals log δ2
j but only Kθ = 2 basis functions on
each off-diagonal component θjl, forcing any fit-
ted cross-power to be smooth and small. H1 thus
occupies the natural middle ground between the
strictly diagonal H0 (off-diagonal data discarded,
no cross-power) and the unrestricted H2:
it sees
the off-diagonal data but is biased toward a near-
diagonal solution, so it can reveal smooth cross-
power when present without introducing spurious
structure when it is not. The posterior spectral ma-
trix is rotated back to the xyz basis for diagnostics.

• H2 (full multivariate xyz). The cross-spectral ma-
trix has fully populated off-diagonals. The multi-
variate Cholesky model (Eqs. 28–30) is fit directly
to the xyz time series with K = 100 basis functions
for every Cholesky component, both diagonal and
off-diagonal. No rotation is needed since the fit is
already in the xyz basis. This is the unrestricted
reference model.

For noise4a the aet basis diagonalises the noise ex-
actly, so the strong diagonal restriction imposed by H0
(and the milder one of H1) are both expected to be
valid descriptions, and all three models should agree

within statistical resolution. For noise5a the per-MOSA
asymmetry introduces genuine off-diagonal power in the
aet basis: H0 is then mis-specified by construction (no
off-diagonal flexibility at all), H1 can absorb only the
small-bandwidth component of the off-diagonals through
its Kθ = 2 basis functions, and H2 remains unre-
stricted. The expected fit-quality ordering is therefore
RISE(H0) > RISE(H1) > RISE(H2) for noise5a and
RISE(H0) (cid:25) RISE(H1) (cid:25) RISE(H2) for noise4a.

For all three models, knots are placed by the adap-
tive quantile rule described in Section II so that more
knots fall near the steep low-frequency rise. We choose
Nb to give blocks of order one week in length, and a fixed
safe-Bayes tempering η = 0.5 (justified by the sensitiv-
ity sweep in Appendix C). Each block is tapered with a
Tukey window (taper fraction 0.1, ENBW (cid:25) 1.04) be-
fore computing the DFT to suppress spectral leakage.
The three TDI transfer-function nulls at 0.030, 0.060,
and 0.090 Hz are excised with half-width 1 mHz prior to
fitting. All other settings follow Section II E.

In addition to the RISE introduced in Section III A,
we report the median relative width of the 90% credible
band on the diagonal PSDs,

∆PSD = medianf

ˆS95%(f ) (cid:0) ˆS5%(f )
ˆS50%(f )

,

(43)

computed in the xyz basis for all three models. This
complements the RISE (a bias metric) with a summary
of posterior precision. Table II reports RISE, ∆PSD, and
wall-clock duration for the three models, two datasets,
and three observation durations. All eighteen LISA runs
satisfy the convergence diagnostics of Section II E: ˆR <
1.01 for all parameters, bulk and tail ESS exceeding 400,
E-BFMI > 0.3, and negligible divergent transitions with
no tree-depth saturation. The results split sharply along
the aet-diagonalisation hypothesis.

Symmetric noise (noise4a). All three models recover
the analytic spectral matrix at the half-percent level
(RISE (cid:25) 4.4 (cid:0) 5.3 (cid:2) 10−3) across all durations and are
statistically indistinguishable from one another. This is
the expected behaviour when the aet basis truly diago-
nalises the noise: H0 already captures the cross-spectral
structure because the only off-diagonal power present in
the xyz basis is the deterministic geometric coupling re-
covered by the M ⊤
(cid:1)Maet rotation, and the additional
aet
off-diagonal flexibility of H1 and H2 has nothing to fit.
Two conclusions follow: (i) for equal-arm LISA noise the
multivariate analysis is not required, and three indepen-
dent univariate fits suﬀice, and (ii) the unrestricted H2
model does not lose accuracy on data that does not need
its flexibility, confirming that H2 remains a safe default
when the diagonal-aet assumption cannot be guaranteed
in advance.

Asymmetric noise (noise5a). The three models sepa-
rate by orders of magnitude in the order predicted by
their off-diagonal flexibility, RISE(H0) > RISE(H1) >
RISE(H2). H0 stalls at RISE (cid:25) 3.3 (cid:2) 10−2 at all du-
rations — a floor set by the aet off-diagonal power it

10

is structurally unable to represent. H1 improves on this
by roughly 50% (RISE (cid:25) 1.8 (cid:0) 2.2 (cid:2) 10−2) by allowing
two off-diagonal basis functions, but is still limited by
its narrow θjk basis. H2 reaches RISE (cid:25) 9(cid:0) 11(cid:2) 10−4,
about 20(cid:2) smaller than H1 and 30 (cid:0) 40(cid:2) smaller than
H0, and continues to improve with duration. This mono-
tone nested ordering is the central diagnostic of the LISA
study: when the per-MOSA noise levels are asymmetric
(the physically realistic regime) the aet rotation does not
diagonalise the cross-spectral matrix, and a multivariate
model that estimates the full Cholesky factor is required.
Posterior uncertainty contracts as expected. For the
multivariate models, the median relative CI width on the
diagonal PSDs falls from (cid:25) 4.7% at 1 month to (cid:25) 1.2%
at 1 year, a reduction of roughly 4(cid:2), consistent with the
p
T scaling expected for variance-dominated Whittle in-
ference, and is essentially identical between H1 and H2
and across datasets, confirming that the per-channel res-
olution is set by the data and not by the choice of basis.
The univariate H0 contracts at the same
T rate but
with consistently narrower bands ((cid:25) 3.4% at 1 month,
(cid:25) 0.9% at 1 year), reflecting the smaller per-channel pa-
rameter count of the p = 1 model when off-diagonals are
removed from the inference. For noise5a this tighter
H0 posterior is a misleading indicator of fit quality —
the bands do not include the truth because the model is
structurally unable to represent the aet off-diagonals, as
the order-of-magnitude RISE gap to H1 and H2 makes
clear.

p

Figure 3 overlays the three durations for both datasets
under the H2 model. The diagonal PSDs are recovered
with sub-percent bias and CI widths that contract with
p
T , while the off-diagonal coherences (the cross-channel
quantities most directly tied to stochastic-background
and parameter-estimation pipelines [35]) are resolved at
the few-percent level by 6 months and tightened by a
further factor of (cid:24) 1.7 at 1 year. A localised feature vis-
ible in all three off-diagonal panels of Fig. 3 is a cluster
of small-amplitude wiggles in the posterior-median co-
herence near the TDI transfer-function null at (cid:25) 0.06 Hz
(and, more weakly, near 0.03 and 0.09 Hz). These arise
because the diagonal PSDs Sii(f ) drop by several orders
of magnitude across each null, and the P-spline basis —
which is smooth on a log scale but agnostic to the loca-
tion and shape of the dips — has to absorb this dynamic
range from the data alone. Small residual misfits at the
bottoms of the dips are amplified in the coherence, pro-
ducing the visible ripple. Incorporating the analytic TDI
transfer functions directly into the prior mean (rather
than only excising a 1 mHz window around each null)
may be a natural way to suppress this artefact and is left
for future work.

IV. DISCUSSION

We have presented a multivariate P-spline framework
for estimating the cross-spectral density matrix of sta-

11

TABLE II. LISA TDI xyz results at fixed safe-Bayes tempering η = 0.5 for the symmetric (noise4a) and asymmetric (noise5a)
datasets, fit with the diagonal-aet null model (H0, off-diagonal Cholesky coeﬀicients fixed to zero), the aet-rotated multivariate
model with restricted off-diagonal basis (H1, Kθ = 2), and the unrestricted multivariate xyz model (H2), at three observation
durations. Nb is the number of Wishart time blocks (chosen to give blocks of approximately seven days in length) and Nc is
set to 1024. RISE is the relative mean squared (Frobenius) error of the posterior-median spectral matrix against the analytic
reference (closed-form equal-link LDC for noise4a, SEGWO with measured per-MOSA ASDs for noise5a), computed in the
xyz basis for all three models (the aet-basis models are first rotated back to xyz via Maet). ∆PSD is the median over frequencies
of the pointwise relative width [ ˆS95%(f ) − ˆS5%(f )]/ ˆS50%(f ) of the 90% credible band on the diagonal PSDs (also in the xyz
basis). Runtime is wall-clock seconds for one full SVI+NUTS pipeline run on a single 4-core CPU node; for H0 this is the total
over the three independent univariate channel fits.
H0

H1

H2

Noise Duration Nb

4a

5a

1 mo
6 mo
1 yr
1 mo
6 mo
1 yr

4
25
52
4
25
52

RISE
4.3×10−3
4.7×10−3
4.4×10−3
3.3×10−2
3.3×10−2
3.3×10−2

∆PSD t (s)

3.4% 16
1.3% 13
0.8% 244
3.5% 15
1.4% 13
1.0% 191

RISE
4.5×10−3
4.7×10−3
4.4×10−3
2.2×10−2
1.8×10−2
2.2×10−2

∆PSD t (s)

4.8% 117
1.9% 224
1.2% 632
4.9% 119
1.9% 218
1.3% 728

RISE
4.7×10−3
5.3×10−3
5.0×10−3
1.1×10−3
9.8×10−4
8.6×10−4

∆PSD

t (s)

4.6% 119
1.9% 181
1.2% 853
4.5% 177
1.9% 245
1.3% 1089

FIG. 3. Posterior summaries of the LISA TDI xyz spectral matrix from the unrestricted H2 model for the symmetric noise4a
(left) and asymmetric noise5a (right) datasets at safe-Bayes tempering η = 0.5, shown at three observation durations: 1 month
(blue), 6 months (orange), and 12 months (green). Diagonal panels show the posterior 90% credible bands for the diagonal
PSDs SXX , SY Y , SZZ . Lower-triangle panels show the coherence |Cij(f )| (Eq. 4) for each pair. Welch reference (grey) is
overplotted.

(cid:97)

tionary vector time series. The Cholesky parametrisa-
tion of S−1 guarantees Hermitian positive definiteness
at every frequency and factorises the likelihood into p
independent per-channel regressions that can be sam-
pled in parallel. A blocked, coarse-grained Whittle likeli-
hood combined with safe-Bayes η-tempering yields near-
nominal 90% credible-interval coverage ((cid:25) 0.90) across a
wide (Nb, Nh) grid on the 3D VAR(2) benchmark, while
reducing the number of inference-time frequency bins by
up to a factor of 32.

On the synthetic 3D VAR(2) benchmark the un-
tempered estimator (η = 1) achieves coverage (cid:25) 0.90 at
every (Nb, Nh) configuration tested. On the LISA TDI

data the same un-tempered estimator over-concentrates.
A fixed safe-Bayes tempering η = 0.5 recovers credible
p
T and remain well-behaved
bands that contract with
across all three observation durations (Appendix C).

a.

η-tempering and debiased Whittle. The η-sweep
of Appendix C shows that credible-band width is a
strong, monotone function of η at fixed (Nb, Nh, K),
but does not yet identify a principled rule for choos-
ing η across regimes. Mapping the optimal η as a func-
tion of NbNh, K, and the underlying spectral curvature
is the most pressing methodological question raised by
this work. A complementary direction is to replace η-
tempering with a debiased Whittle likelihood [36], which

101410121010PSD [1/Hz]SXX0.000.250.500.751.00|Cij(f)||CYX|101410121010SYY104103102101Frequency [Hz]0.000.250.500.751.00|Cij(f)||CZX|104103102101Frequency [Hz]0.000.250.500.751.00|CZY|104103102101Frequency [Hz]101410121010SZZWelch reference1 month 90% CI6 months 90% CI12 months 90% CI101410121010PSD [1/Hz]SXX0.000.250.500.751.00|Cij(f)||CYX|101410121010SYY104103102101Frequency [Hz]0.000.250.500.751.00|Cij(f)||CZX|104103102101Frequency [Hz]0.000.250.500.751.00|CZY|104103102101Frequency [Hz]101410121010SZZWelch reference1 month 90% CI6 months 90% CI12 months 90% CIb.

corrects the finite-sample bias of the periodogram analyt-
ically rather than down-weighting the entire likelihood.
Informative priors. The hierarchical Gaussian
smoothing prior used here is near-uninformative: each
weight has marginal variance set by a high-level hyper-
parameter (ϕj, νj) with no scale imposed by the data
ahead of inference. In practice this places no bound on
where the spline curves can sit, and most of NUTS warm-
up is spent locating a region of weight space consistent
with even a coarse Welch estimate. A more informative
prior, centred on a fast initial estimate of log δ2
j (f ) and
the off-diagonal Cholesky terms (e.g. from a Welch-and-
smooth pre-pass), with ϕj controlling allowed deviations
from that estimate, may (i) make the smoothing hyper-
parameters physically meaningful as fractional excursion
bounds, (ii) shorten warm-up substantially, and (iii) re-
duce the prior–data contention that drives over-coverage
at small η in the appendix sweep.

c. Non-stationarity. The Whittle

likelihood as-
Both noise4a and
sumes wide-sense stationarity.
noise5a are stationary by construction: realisations of
an LDC noise model with constant per-MOSA amplitude
spectral densities and constant 8.3 s light travel times.
The real LISA noise environment is non-stationary on
multiple timescales — constellation breathing modulates
the arm lengths over the year-long orbital period, and
per-link noise levels drift on timescales from hours (ther-
mal cycling, glitches) to months (component aging). A
natural and tractable extension is to assume piecewise
stationarity: partition the mission timeline into segments
short enough that within-segment stationarity is a good
approximation but long enough that the Whittle approx-
imation remains accurate, and either fit each segment in-
dependently or couple consecutive segments through a
smoothness prior on the frequency-dependent Cholesky
parameters in the time direction. This would extend the
present model to a time-frequency spectral matrix S(t, f )
at the cost of one additional smoothing dimension.

d.

Improved variational

families. As reported in
Section III A, the current low-rank (r = 16) multivariate
Gaussian SVI guide recovers an accurate posterior me-
dian but produces narrower credible bands than NUTS,
with a corresponding drop in empirical coverage. NUTS
is therefore retained as the production sampler, and SVI
is used only as an initialiser. Because the model is im-
plemented in NumPyro, the variational family is a swap-
pable component: replacing the Gaussian guide with a
normalising flow (e.g. a neural spline flow or a masked au-
toregressive / inverse autoregressive flow) is essentially a
one-line change. A systematic study of which variational
families recover calibrated PSD posteriors — so that SVI
could serve as a stand-alone approximation at a fraction
of the NUTS wall-clock cost — is left for future work.

12

metric/nonparametric models that embed known instru-
mental features within the P-spline envelope are another
route to improving sharp-feature accuracy without sacri-
ficing the smoothness elsewhere.

Data and Software Availability

The software developed for this project is open-source
and publicly available from the GitHub repository
https://github.com/nz-gravity/LogPSplinePSD.
git, which contains all source code, example scripts,
and configuration files needed to reproduce the results.
Installation instructions and dependency information
(tested on Python (cid:21) 3.10) are provided in the repository
README. The in-text code links ((cid:97)) are pinned to the
tagged release v0.1.0 of the package, so that the
referenced line numbers remain valid as the codebase
evolves. The software is released under the MIT License,
permitting free use, modification, and distribution. The
LISA time series data are available at [28, 29]. Scripts
for the LISA analysis are available from the GitHub
repository
https://github.com/nz-gravity/lisa_
multivar_logpsplines_study.git.

ACKNOWLEDGMENTS

We thank Quentin Baghi, Jean-Baptiste Bayle, Ollie
Burke, Nikolaos Karnesis, Martina Muratore, Federico
Pozzoli, and others in the LISA Noise Non-Stationarities
Group (part of the “Deep analysis group” of the Dis-
tributed Data Processing Centre, DDPC) for helpful dis-
cussions. We additionally thank Jean-Baptiste Bayle
for providing the LISA dataset, and the University of
Glasgow for the computing resources that supported JB
in simulating the dataset. We acknowledge the use
of the SEGWO package [34], developed by J.-B. Bayle
and O. Hartwig, used here to construct the analytic
3(cid:2)3 cross-spectral reference for the asymmetric noise5a
realisation. PMR, RM, and AV gratefully acknowl-
edge support from the Marsden Fund Council grants
MFP-UOA2131 and MFP-UOA2531, funded by the New
Zealand Government and managed by the Royal Society
Te Apārangi. This work was performed on the OzSTAR
national facility at Swinburne University of Technology.
The OzSTAR program receives funding in part from the
Astronomy National Collaborative Research Infrastruc-
ture Strategy (NCRIS) allocation provided by the Aus-
tralian Government, and from the Victorian Higher Ed-
ucation State Investment Fund (VHESIF) provided by
the Victorian Government.

e. Other extensions. The within-bin constant-S ap-
proximation underlying coarse-graining (Eq. 13) can
bias estimates near sharp spectral
features. Adap-
tive bin widths that narrow around lines and transfer-
function nulls are a natural extension. Hybrid para-

Appendix A: Notation

Table III lists the symbols used throughout the paper,
grouped by the stage of the model in which they appear.

Symbol

Meaning

13

T, ∆t, fs
fN y
n, p
∆f , fk
N

Nb, Tb
Nh, Jh, ¯fh

Nc

Z
d(fk)
Γ(τ ), γlm(τ )
S(fk)

I(i)(fk), ¯I(fk)
Y(fk)
Ycg
h

Time series & sampling

Total observation duration, sampling interval, sampling frequency; T = n∆t, ∆t = 1/fs.
Nyquist frequency, fN y = fs/2.
Number of time samples per channel, number of channels.
Frequency resolution ∆f = 1/T ; kth DFT frequency fk = k∆f .
Number of positive Fourier frequencies, N = ⌊n/2⌋.

Blocking & coarse-graining

Number of non-overlapping time blocks; block duration Tb = T /Nb (n/Nb samples per block).
Number of fine frequencies aggregated per coarse bin; Jh the hth bin (consecutive subset of fine
frequencies), with midpoint ¯fh.
Number of coarse-grain bins, Nc = n/(2NbNh).

Spectral density

Observed p-channel time series, Z ∈ Rn×p.
DFT vector, d(fk) ∈ Cp.
Autocovariance matrix at integer lag τ and its (l, m) entry.
p × p Hermitian positive-definite spectral density matrix; Fourier transform of Γ.

Periodogram & Wishart statistics

Block-i periodogram, I(i) = d(i)d(i)∗ ∼ CW p(TbS, 1); block-averaged periodogram ¯I = N
Summed periodogram / Wishart statistic, Y = Nb
Coarse-grained statistic, Ycg

¯I ∼ CW p(TbS, Nb).

∑

h =

fk∈Jh

Y(fk) ˙∼ CW p(TbS( ¯fh), NbNh).
√

∑

−1
b

i I(i).

ν , v(k)
λ(k)

ν , u(k)

ν

νth eigenvalue/unit eigenvector of Y(fk); scaled eigenvector u(k)

ν =

ν v(k)
λ(k)
ν .

Tk, Dk
θ(k)
jl , δ2
jk

θj , δj

K, Bm(fk)

Kj , Kjl
w(δ)
j

, w(ℜ)
jl

, w(ℑ)
jl

Pj , ϕj

Factors in S(fk)−1 = T∗

kD

Cholesky parametrisation
−1
k Tk: Tk unit lower-triangular, Dk diagonal with entries δ2

1k, . . . , δ2

pk.

Off-diagonal entry (j, l) of Tk (j > l) and jth diagonal entry of Dk.
Parameter vectors for channel j: collect θ(h)
jl

(l < j) and δjh across the coarse bins h = 1, . . . , Nc.

P-spline model

Number of B-spline basis functions (model hyperparameter, distinct from frequency index k); mth basis
evaluated at fk.
Basis sizes for the jth diagonal and (j, l) off-diagonal components.

Spline coeﬀicient vectors for log δ2
l B′′

Penalty matrix [Pj ]lm =

0 B′′

∫

1

jk, ℜ[θjl], ℑ[θjl].
m dt and smoothing precision hyperparameter.

TABLE III. Notation used throughout the paper. Symbols are grouped by the stage of the model they belong to.

Appendix B: Bivariate VAR(2) benchmark

This appendix benchmarks the proposed estimator
against the bivariate Bayesian baselines available in the
literature on a controlled VAR(2) problem with closed-
form ground truth. Following Liu et al. [20], we simu-
late 500 independent realisations of a bivariate VAR(2)
time series at three sample sizes n = 256, 512, 1024 (re-
fer to Liu et al. [Section 4.2, 20] for the definitions of
the VAR(2) models), and the theoretical spectral density
matrix provides exact ground truth. The spectral densi-
ties are estimated using the multivariate P-spline method

with K = 16 basis functions and degree-2 splines (d = 2)
for each spectral component. We compare to the varia-
tional Bayes (VB) and variational nonparametric correc-
tion (VNPC) methods of Liu et al. [20] using the same
L2 error criterion. The L2 error is the un-normalised
integrated Frobenius error of the posterior-median spec-
kS(f )kF df . We use
tral matrix, without dividing by
it here for direct comparison with the VB and VNPC
results, which are reported in this metric.

R

Table IV summarises the L2 errors across the 500 inde-
pendent realisations. The multivariate P-spline estima-
tor is competitive with both comparators at every sam-

TABLE IV. Bivariate VAR(2) benchmark using the L2 error
metric from Liu et al. [20]. Entries are mean ± standard
deviation over 500 independent realisations. Smaller values
are better.

n

P-spline

VB

VNPC

0.095 ± 0.022 0.121 ± 0.029 0.129 ± 0.033
256
0.072 ± 0.018 0.091 ± 0.027 0.103 ± 0.025
512
1024 0.055 ± 0.013 0.066 ± 0.016 0.082 ± 0.017

ple size and attains the lowest mean error in all three set-
tings. The mean L2 error decreases from 0.095 at n = 256
to 0.072 at n = 512 and 0.055 at n = 1024, compared
with 0.121, 0.091, and 0.066 for VB and 0.129, 0.103,
and 0.082 for VNPC. This is the main message we want
from the bivariate benchmark: the proposed multivariate
P-spline method achieves accuracy of the same order as
established alternatives, while in these simulations being
modestly more accurate on average.

Appendix C: η-tempering sensitivity on noise4a

To support the choice of tempering parameter for
the LISA application, we re-ran the noise4a dataset at

14

η 2 f0.01, 0.03, 0.1, 0.25, 0.5, 0.75, 1g for each of the three
observation durations (1, 6, and 12 months) used in Sec-
tion III B, holding all other settings fixed. Figure 4 sum-
marises the result.

Two features emerge from Figure 4. First, the matrix
RISE is essentially flat across the entire sweep: it varies
by less than (cid:6)5% about its mean at every duration, con-
firming that the posterior-median point estimate is nearly
insensitive to η. Second, credible-band width contracts
monotonically with η, by roughly an order of magnitude
from η = 0.01 to η = 1, with the steepest contraction at
the longest duration—consistent with the per-bin Fisher
information scaling as NbNh (Eq. (27)).

The vertical dotted line marks the choice η = 0.5
used in Section III B. This value gives a practical com-
promise: credible bands are meaningfully wide at all
three durations while the point estimate remains accu-
rate throughout. Reaching nominal coverage would re-
quire yet smaller η or a more flexible model (e.g. adaptive
coarse-graining near the TDI nulls). We leave this to fu-
ture work.

[1] T. B. Littenberg and N. J. Cornish, Bayesian inference for
spectral estimation of gravitational wave detector noise,
Physical Review D 91, 084034 (2015).

[2] P. A.-S. et al., Laser interferometer space antenna (2017),

arXiv:1702.00786.

[3] C. Kirch, M. C. Edwards, A. Meier, and R. Meyer, Be-
yond Whittle: Nonparametric Correction of a Parametric
Likelihood with a Focus on Bayesian Time Series Analy-
sis, Bayesian Analysis 14, 1037 (2019).

[4] J. Liu, A. Vajpeyi, R. Meyer, K. Janssens, J. E. Lee,
et al., Variational inference for correlated gravitational
wave detector network noise, Phys. Rev. D 111, 062003
(2025).

[5] M. C. Edwards, R. Meyer, and N. Christensen, Bayesian
nonparametric spectral density estimation using B-spline
priors, Statistics and Computing 29, 67 (2019).

[6] P. H. C. Eilers and B. D. Marx, Flexible smoothing with
B-splines and penalties, Statistical Science 11, 89 (1996).
[7] P. Maturana-Russel and R. Meyer, Bayesian spec-
tral density estimation using P-splines with quantile-
based knot placement, Computational Statistics 36, 2055
(2021).

[8] N. Aimen, P. Maturana-Russel, A. Vajpeyi, N. Chris-
tensen, and R. Meyer, Bayesian power spectral density es-
timation for LISA noise based on penalized splines with a
parametric boost, Physical Review D 113, 024022 (2026).
[9] Q. Baghi, N. Karnesis, J.-B. Bayle, M. Besançon,
and H. Inchauspé, Uncovering gravitational-wave back-
grounds from noises of unknown shape with LISA, Jour-
nal of Cosmology and Astroparticle Physics 2023 (04),
066.

[10] A. Santini, M. Muratore, J. Gair, and O. Hartwig, Flex-
ible, GPU-accelerated approach for the joint character-
ization of LISA instrumental noise and stochastic grav-
itational wave backgrounds, Phys. Rev. D 112, 084050
(2025), arXiv:2507.06300 [gr-qc].

[11] M. Muratore, Instrumental modelling and noise reduction
algorithms for the Laser Interferometer Space Antenna,
Ph.D. thesis, Gottfried Wilhelm Leibniz Universität Han-
nover (2021).

[12] F. Cireddu, M. Wils, I. C. F. Wong, P. T. H. Pang,
T. G. F. Li, et al., Likelihood for a network of
gravitational-wave detectors with correlated noise, Phys.
Rev. D 110, 104060 (2024).

[13] O. Hartwig, M. Lilley, M. Muratore, and M. Pieroni,
Stochastic gravitational wave background reconstruction
for a nonequilateral and unequal-noise lisa constellation,
Phys. Rev. D 107, 123531 (2023).

[14] M. Muratore, J. Gair, and L. Speri, Impact of the noise
knowledge uncertainty for the science exploitation of cos-
mological and astrophysical stochastic gravitational wave
background with lisa, Phys. Rev. D 109, 042001 (2024).
[15] O. Rosen and D. S. Stoffer, Automatic estimation of mul-
tivariate spectra via smoothing splines, Biometrika 94,
335 (2007).

[16] Z. Hu and R. Prado, Fast Bayesian inference on spectral
analysis of multivariate stationary time series, Computa-
tional Statistics & Data Analysis 178, 107596 (2023).
[17] P. Grünwald and T. van Ommen, Inconsistency of
Bayesian inference for misspecified linear models, and
a proposal for repairing it, Bayesian Analysis 12, 1069
(2017).

15

proach to estimating directional ocean wave models,
Journal of the Royal Statistical Society Series C: Applied
Statistics 72, 544 (2023).

[22] M. P. Wand and J. T. Ormerod, On semiparametric re-
gression with o’sullivan penalized splines, Australian &
New Zealand Journal of Statistics 50, 179 (2008).
[23] J. Bradbury, R. Frostig, P. Hawkins, M. J. Johnson,
C. Leary, et al., JAX: composable transformations of
Python+NumPy programs (2018).

[24] D. Phan, N. Pradhan, and M. Jankowiak, Composable
effects for flexible and accelerated probabilistic program-
ming in pyro, arXiv preprint arXiv:1912.11554 (2019).

[25] D. P. Kingma and J. Ba, Adam: A method for stochastic
optimization, in 3rd International Conference on Learn-
ing Representations
(ICLR) (2015) arXiv:1412.6980
[cs.LG].

[26] A. Gelman and D. B. Rubin, Inference from Iterative
Simulation Using Multiple Sequences, Statistical Science
7, 457 (1992).

[27] R. Kumar, C. Carroll, A. Hartikainen, and O. Mar-
tin, Arviz a unified library for exploratory analysis of
bayesian models in python, Journal of Open Source Soft-
ware 4, 1143 (2019).

[28] J.-B. Bayle, LISA SGWB Dataset (noise-4a) (2025),
https://zenodo.org/doi/10.5281/zenodo.15698080.
[29] J.-B. Bayle, Lisa sgwb dataset (noise-5a), To be assigned

(2026).

[30] J.-B. Bayle, O. Hartwig, and M. Staab, Lisa instrument

(2024).

(cid:97)

[31] M. Le Jeune, S. Babak, Q. Baghi, J.-B. Bayle, E. Castelli,
and N. Korsakova, Lisa data challenge spritz (ldc2b),
10.5281/zenodo.7436568 (2022).

[32] M. Staab, J.-B. Bayle, and O. Hartwig, PyTDI (2025),

https://zenodo.org/doi/10.5281/zenodo.6351736.

[33] T. A. Prince, M. Tinto, S. L. Larson, and J. W. Arm-
strong, LISA optimal sensitivity, Phys. Rev. D 66,
122002 (2002), arXiv:gr-qc/0209039 [gr-qc].

[34] J.-B. Bayle and O. Hartwig, SEGWO: Sensitivity esti-

mator for gravitational-wave observatories.

[35] M. L. Katz et al., The LISA global fit, arXiv preprint

arXiv:2404.12571 (2024).

[36] A. M. Sykulski, S. C. Olhede, A. P. Guillaumin, J. M.
Lilly, and J. J. Early, The debiased Whittle likelihood,
Biometrika 106, 251 (2019).

FIG. 4.
Safe-Bayes η-sensitivity on the symmetric LISA
noise4a dataset at three observation durations (1 month,
6 months, 1 year). Top: median pointwise relative width of
the 90% credible band on the diagonal PSDs. Bottom: ma-
trix RISE (narrow y-range—RISE varies by ≲5% across the
full sweep). The vertical dotted line marks η = 0.5, the value
used in Section III B.

[18] M. D. Hoffman, D. M. Blei, C. Wang, and J. Pais-
ley, Stochastic variational inference, Journal of machine
learning research (2013).

[19] M. D. Hoffman, A. Gelman, et al., The no-u-turn sam-
pler: adaptively setting path lengths in hamiltonian
monte carlo., J. Mach. Learn. Res. 15, 1593 (2014).
[20] Y. Liu, C. Kirch, J. E. Lee, and R. Meyer, A nonparamet-
rically corrected likelihood for bayesian spectral analysis
of multivariate time series (2024).

[21] J. P. Grainger, A. M. Sykulski, K. Ewans, H. F. Hansen,
and P. Jonathan, A multivariate pseudo-likelihood ap-

0.00.10.20.3PSD1 month6 months1 year1021011000.00450.00500.0055RISE
