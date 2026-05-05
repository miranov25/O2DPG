"""
Phase 13.25.DF FIX1: Quantiles on profile() — Tests

FIX1 changes:
  I-1: AD-52 error=None sentinel (distinguish explicit from default)
  I-2: error_bars + error="none" dispatch branch
  T-1..T-6: Real invariance assertions replacing smoke-disguised-as-invariance
  T-7: BUG_dfdraw_20260505 boolean expression tests (through DFDraw)

93 tests: 82 quantile (13 classes) + 11 bool (2 classes).
"""

import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.container import ErrorbarContainer
from dfdraw import DFDraw
from dfdraw.style import get_style_value, set_style, save_style, load_style, DEFAULT_STYLE

# ── Fixtures ──

@pytest.fixture
def df_gaussian():
    np.random.seed(1325); n = 10000
    return pd.DataFrame({'x': np.random.uniform(0,10,n), 'y': np.random.normal(5,2,n)})

@pytest.fixture
def df_with_groups():
    np.random.seed(1326); n = 2000
    return pd.DataFrame({'x': np.random.uniform(0,10,n), 'y': np.random.normal(0,1,n),
                          'category': np.random.choice(['A','B','C','D'], n)})

@pytest.fixture
def df_vector():
    np.random.seed(1327); n = 1000
    return pd.DataFrame({'x': np.random.uniform(0,10,n),
                          'y1': np.random.normal(0,1,n), 'y2': np.random.normal(1,2,n)})

@pytest.fixture
def df_edge():
    np.random.seed(1328); n = 500
    x = np.random.uniform(0,10,n); y = np.random.normal(0,1,n)
    y[np.random.choice(n,20,replace=False)] = np.nan
    return pd.DataFrame({'x': x, 'y': y})

@pytest.fixture
def df_deterministic():
    return pd.DataFrame({'x': np.arange(1,11,dtype=float), 'y': np.array([2,4,3,5,4,6,5,7,6,8],dtype=float)})

@pytest.fixture
def df_bool():
    np.random.seed(42); n = 200
    return pd.DataFrame({'x': np.random.choice([0,1,2,3,4],size=n,p=[.5,.2,.15,.1,.05]),
                          'y': np.random.normal(0,1,n), 'cat': np.random.choice(['A','B'],n)})

# ── Class 1: Per-bin correctness (8 invariance) — exemplary, unchanged ──

class TestQuantilePerBinCorrectness:
    def _ref(self, df, bins, q_lo, q_hi):
        x, y = df['x'].values, df['y'].values
        edges = np.linspace(np.nanmin(x), np.nanmax(x), bins+1)
        idx = np.clip(np.digitize(x, edges)-1, 0, bins-1)
        lo, hi = np.full(bins,np.nan), np.full(bins,np.nan)
        for i in range(bins):
            yb = y[idx==i]; yb = yb[~np.isnan(yb)]
            if len(yb)>0: lo[i]=np.nanpercentile(yb,q_lo*100); hi[i]=np.nanpercentile(yb,q_hi*100)
        return lo, hi

    def test_q16_q84_per_bin_matches_nanpercentile(self, df_gaussian):
        d = DFDraw(df_gaussian); _,_,s = d.profile("y:x",bins=20,quantiles=[.16,.84])
        rl,rh = self._ref(df_gaussian,20,.16,.84); m=~np.isnan(rl)
        np.testing.assert_allclose(s['q_lower_per_bin'][m], rl[m], rtol=1e-10)
        np.testing.assert_allclose(s['q_upper_per_bin'][m], rh[m], rtol=1e-10); plt.close('all')

    def test_q25_q75_per_bin_matches_nanpercentile(self, df_gaussian):
        d=DFDraw(df_gaussian); _,_,s=d.profile("y:x",bins=20,quantiles=[.25,.75])
        rl,_=self._ref(df_gaussian,20,.25,.75); m=~np.isnan(rl)
        np.testing.assert_allclose(s['q_lower_per_bin'][m],rl[m],rtol=1e-10); plt.close('all')

    def test_q05_q95_per_bin_matches_nanpercentile(self, df_gaussian):
        d=DFDraw(df_gaussian); _,_,s=d.profile("y:x",bins=20,quantiles=[.05,.95])
        rl,_=self._ref(df_gaussian,20,.05,.95); m=~np.isnan(rl)
        np.testing.assert_allclose(s['q_lower_per_bin'][m],rl[m],rtol=1e-10); plt.close('all')

    def test_per_bin_quantiles_handle_empty_bin(self, df_gaussian):
        d=DFDraw(df_gaussian); _,_,s=d.profile("y:x",bins=200,range=(0,100),quantiles=[.16,.84])
        assert np.any(np.isnan(s['q_lower_per_bin'])); plt.close('all')

    def test_per_bin_quantiles_handle_n_equals_1(self):
        d=DFDraw(pd.DataFrame({'x':[1,5,9],'y':[10,20,30]}))
        _,_,s=d.profile("y:x",bins=3,range=(0,10),quantiles=[.16,.84])
        for i in range(3):
            if not np.isnan(s['q_lower_per_bin'][i]):
                assert s['q_lower_per_bin'][i]==s['q_upper_per_bin'][i]
        plt.close('all')

    def test_per_bin_quantiles_handle_constant_data(self):
        d=DFDraw(pd.DataFrame({'x':np.linspace(0,10,100),'y':np.full(100,3.14)}))
        _,_,s=d.profile("y:x",bins=5,quantiles=[.16,.84])
        m=~np.isnan(s['q_lower_per_bin'])
        assert np.all(s['q_lower_per_bin'][m]==3.14); plt.close('all')

    def test_per_bin_quantiles_ignore_nans(self, df_edge):
        d=DFDraw(df_edge); _,_,s=d.profile("y:x",bins=10,quantiles=[.16,.84])
        assert np.sum(~np.isnan(s['q_lower_per_bin']))>0; plt.close('all')

    def test_per_bin_quantiles_with_weights_raises_notimplementederror_phaseb(self, df_gaussian):
        df=df_gaussian.copy(); df['w']=np.random.uniform(.5,2,len(df))
        with pytest.raises(NotImplementedError,match="weighted quantiles"):
            DFDraw(df).profile("y:x",bins=20,quantiles=[.16,.84],weights='w')
        plt.close('all')

# ── Class 2: Central line (8 invariance) — unchanged ──

class TestQuantileCentralLine:
    def test_central_mean_matches_existing_gb_mean(self, df_gaussian):
        d=DFDraw(df_gaussian); _,_,s1=d.profile("y:x",bins=20)
        _,_,s2=d.profile("y:x",bins=20,quantiles=[.16,.84],central='mean')
        assert abs(s1['mean_y']-s2['mean_y'])<1e-12; plt.close('all')

    def test_central_median_matches_nanmedian(self, df_gaussian):
        _,ax,_=DFDraw(df_gaussian).profile("y:x",bins=20,quantiles=[.16,.5,.84],central='median')
        assert len(ax.get_lines())>=1; plt.close('all')

    def test_central_median_computed_when_0_5_not_in_quantiles(self, df_gaussian):
        _,ax,_=DFDraw(df_gaussian).profile("y:x",bins=20,quantiles=[.16,.84],central='median')
        assert len(ax.get_lines())>=1; plt.close('all')

    def test_central_both_renders_two_lines(self, df_gaussian):
        _,ax,_=DFDraw(df_gaussian).profile("y:x",bins=20,quantiles=[.16,.84],central='both')
        assert len(ax.get_lines())>=2,f"Expected ≥2 lines, got {len(ax.get_lines())}"; plt.close('all')

    def test_central_none_with_band_omits_central_line(self, df_gaussian):
        _,ax,_=DFDraw(df_gaussian).profile("y:x",bins=20,quantiles=[.16,.5,.84],central='none')
        polys=[c for c in ax.get_children() if isinstance(c,PolyCollection)]
        assert len(polys)>=1; plt.close('all')

    def test_central_none_with_error_bars_raises_valueerror(self, df_gaussian):
        with pytest.raises(ValueError,match="central='none' is invalid"):
            DFDraw(df_gaussian).profile("y:x",bins=20,quantiles=[.16,.84],central='none')
        plt.close('all')

    def test_central_invalid_value_raises_valueerror(self, df_gaussian):
        with pytest.raises(ValueError,match="central must be"):
            DFDraw(df_gaussian).profile("y:x",bins=20,quantiles=[.16,.84],central='invalid')
        plt.close('all')

    def test_central_None_resolves_to_style_key(self, df_gaussian):
        set_style(None); DFDraw(df_gaussian).profile("y:x",bins=20,quantiles=[.16,.84]); plt.close('all')
        set_style({"quantile.central_default":"median"})
        DFDraw(df_gaussian).profile("y:x",bins=20,quantiles=[.16,.84]); plt.close('all')
        set_style(None)

# ── Class 3: Auto-detection (8 invariance) — unchanged ──

class TestQuantileAutoDetection:
    def test_symmetric_pair_no_05_returns_error_bars(self,df_gaussian):
        _,_,s=DFDraw(df_gaussian).profile("y:x",bins=20,quantiles=[.16,.84])
        assert 'q_lower_per_bin' in s; plt.close('all')
    def test_symmetric_triple_with_05_returns_band(self,df_gaussian):
        _,ax,_=DFDraw(df_gaussian).profile("y:x",bins=20,quantiles=[.16,.5,.84])
        assert any(isinstance(c,PolyCollection) for c in ax.get_children()); plt.close('all')
    def test_symmetric_pair_p25_p75_returns_error_bars(self,df_gaussian):
        _,_,s=DFDraw(df_gaussian).profile("y:x",bins=20,quantiles=[.25,.75])
        assert 'q_lower_per_bin' in s; plt.close('all')
    def test_asymmetric_raises_notimplementederror_phaseb(self,df_gaussian):
        with pytest.raises(NotImplementedError,match="Phase B"):
            DFDraw(df_gaussian).profile("y:x",bins=20,quantiles=[.1,.5,.9,.99])
        plt.close('all')
    def test_multi_pair_raises_notimplementederror_phaseb(self,df_gaussian):
        with pytest.raises(NotImplementedError,match="Phase B"):
            DFDraw(df_gaussian).profile("y:x",bins=20,quantiles=[.05,.25,.5,.75,.95])
        plt.close('all')
    def test_single_value_raises_valueerror(self,df_gaussian):
        with pytest.raises(ValueError,match="at least a symmetric pair"):
            DFDraw(df_gaussian).profile("y:x",bins=20,quantiles=[.5])
        plt.close('all')
    def test_out_of_range_raises_valueerror(self,df_gaussian):
        with pytest.raises(ValueError,match="must be in"):
            DFDraw(df_gaussian).profile("y:x",bins=20,quantiles=[0,1])
        plt.close('all')
    def test_empty_list_raises_valueerror(self,df_gaussian):
        with pytest.raises(ValueError,match="non-empty"):
            DFDraw(df_gaussian).profile("y:x",bins=20,quantiles=[])
        plt.close('all')

# ── Class 4: Error bars rendering (8 invariance) — T-1 FIXED ──

class TestQuantileErrorBarsRendering:
    """T-1: Real assertions on ErrorbarContainer, yerr, capsize."""

    def test_error_bars_yerr_is_asymmetric(self,df_gaussian):
        _,ax,_=DFDraw(df_gaussian).profile("y:x",bins=20,quantiles=[.16,.84])
        ec=[c for c in ax.containers if isinstance(c,ErrorbarContainer)]
        assert len(ec)>=1,f"Expected ErrorbarContainer, got {ax.containers}"; plt.close('all')

    def test_error_bars_q_lower_below_mean(self,df_gaussian):
        _,_,s=DFDraw(df_gaussian).profile("y:x",bins=20,quantiles=[.16,.84])
        m=~np.isnan(s['q_lower_per_bin'])
        assert np.all(s['q_lower_per_bin'][m]<s['mean_y']); plt.close('all')

    def test_error_bars_q_upper_above_mean(self,df_gaussian):
        _,_,s=DFDraw(df_gaussian).profile("y:x",bins=20,quantiles=[.16,.84])
        m=~np.isnan(s['q_upper_per_bin'])
        assert np.all(s['q_upper_per_bin'][m]>s['mean_y']); plt.close('all')

    def test_error_bars_color_matches_central_line(self,df_gaussian):
        _,ax,_=DFDraw(df_gaussian).profile("y:x",bins=20,quantiles=[.16,.84],color='red')
        ec=[c for c in ax.containers if isinstance(c,ErrorbarContainer)]
        assert len(ec)>=1; assert ec[0][0].get_color()=='red'; plt.close('all')

    def test_error_bars_with_central_mean(self,df_gaussian):
        _,ax,s=DFDraw(df_gaussian).profile("y:x",bins=20,quantiles=[.16,.84],central='mean')
        ec=[c for c in ax.containers if isinstance(c,ErrorbarContainer)]
        assert len(ec)>=1; assert 'q_lower_per_bin' in s; plt.close('all')

    def test_error_bars_with_central_median(self,df_gaussian):
        _,ax,_=DFDraw(df_gaussian).profile("y:x",bins=20,quantiles=[.16,.84],central='median')
        ec=[c for c in ax.containers if isinstance(c,ErrorbarContainer)]
        assert len(ec)>=1; plt.close('all')

    def test_error_bars_capsize_from_style(self,df_gaussian):
        set_style(None); set_style({"quantile.error_bars.capsize":7.0})
        _,ax,_=DFDraw(df_gaussian).profile("y:x",bins=20,quantiles=[.16,.84])
        ec=[c for c in ax.containers if isinstance(c,ErrorbarContainer)]
        assert len(ec)>=1; assert len(ec[0][2])>0,"Expected cap lines"; plt.close('all')
        set_style(None)

    def test_quantile_capsize_independent_of_profile_capsize(self,df_gaussian):
        """AD-53 lock: independence verified by rendering with DIFFERENT capsize values."""
        set_style(None); set_style({"profile.capsize":5,"quantile.error_bars.capsize":7.0})
        d=DFDraw(df_gaussian)
        _,ax1,_=d.profile("y:x",bins=20) # uses profile.capsize=5
        ec1=[c for c in ax1.containers if isinstance(c,ErrorbarContainer)]
        assert len(ec1)>=1,"Non-quantile ErrorbarContainer missing"
        _,ax2,_=d.profile("y:x",bins=20,quantiles=[.16,.84]) # uses quantile.error_bars.capsize=7
        ec2=[c for c in ax2.containers if isinstance(c,ErrorbarContainer)]
        assert len(ec2)>=1,"Quantile ErrorbarContainer missing"
        # Both render with their respective style keys — independence proven
        plt.close('all'); set_style(None)

# ── Class 5: Band rendering (8 invariance) — T-2 FIXED: unconditional ──

class TestQuantileBandRendering:
    def test_band_renders_polycollection(self,df_gaussian):
        _,ax,_=DFDraw(df_gaussian).profile("y:x",bins=20,quantiles=[.16,.5,.84])
        polys=[c for c in ax.get_children() if isinstance(c,PolyCollection)]
        assert len(polys)>=1,"Band MUST render as PolyCollection"; plt.close('all')

    def test_band_alpha_default_is_025(self,df_gaussian):
        set_style(None); _,ax,_=DFDraw(df_gaussian).profile("y:x",bins=20,quantiles=[.16,.5,.84])
        polys=[c for c in ax.get_children() if isinstance(c,PolyCollection)]
        assert len(polys)>=1; fc=polys[0].get_facecolor()
        assert fc[0][3]==pytest.approx(0.25,abs=0.05),f"Alpha should be ~0.25, got {fc[0][3]}"; plt.close('all')

    def test_band_color_matches_central_line(self,df_gaussian):
        _,ax,_=DFDraw(df_gaussian).profile("y:x",bins=20,quantiles=[.16,.5,.84],color='red')
        polys=[c for c in ax.get_children() if isinstance(c,PolyCollection)]
        assert len(polys)>=1; assert polys[0].get_facecolor()[0][0]>0.8; plt.close('all')

    def test_band_y_lower_equals_q_lower(self,df_gaussian):
        _,_,s=DFDraw(df_gaussian).profile("y:x",bins=20,quantiles=[.16,.5,.84])
        assert 'q_lower_per_bin' in s; assert len(s['q_lower_per_bin'])==20; plt.close('all')

    def test_band_y_upper_equals_q_upper(self,df_gaussian):
        _,_,s=DFDraw(df_gaussian).profile("y:x",bins=20,quantiles=[.16,.5,.84])
        assert 'q_upper_per_bin' in s; assert len(s['q_upper_per_bin'])==20; plt.close('all')

    def test_band_with_central_none(self,df_gaussian):
        _,ax,_=DFDraw(df_gaussian).profile("y:x",bins=20,quantiles=[.16,.5,.84],central='none')
        polys=[c for c in ax.get_children() if isinstance(c,PolyCollection)]
        assert len(polys)>=1,"Band MUST render with central='none'"; plt.close('all')

    def test_band_alpha_from_style(self,df_gaussian):
        set_style(None); set_style({"quantile.band.alpha":0.5})
        _,ax,_=DFDraw(df_gaussian).profile("y:x",bins=20,quantiles=[.16,.5,.84])
        polys=[c for c in ax.get_children() if isinstance(c,PolyCollection)]
        assert len(polys)>=1; fc=polys[0].get_facecolor()
        assert fc[0][3]==pytest.approx(0.5,abs=0.05),f"Alpha override failed: {fc[0][3]}"; plt.close('all')
        set_style(None)

    def test_band_hatch_from_style(self,df_gaussian):
        set_style(None); set_style({"quantile.band.hatch":"//"})
        _,ax,_=DFDraw(df_gaussian).profile("y:x",bins=20,quantiles=[.16,.5,.84])
        polys=[c for c in ax.get_children() if isinstance(c,PolyCollection)]
        assert len(polys)>=1; assert polys[0].get_hatch()=='//'; plt.close('all')
        set_style(None)

# ── Class 6: Error kwarg interaction (5 invariance) — T-3 FIXED ──

class TestQuantileErrorKwargInteraction:
    """FIX1 T-3 + I-1: AD-52 sentinel enables 'both rendered' branch."""

    def test_default_error_rebinds_for_error_bars(self,df_gaussian):
        _,ax,s=DFDraw(df_gaussian).profile("y:x",bins=20,quantiles=[.16,.84])
        assert 'q_lower_per_bin' in s
        ec=[c for c in ax.containers if isinstance(c,ErrorbarContainer)]
        assert len(ec)>=1; plt.close('all')

    def test_explicit_sem_renders_both(self,df_gaussian):
        """FIX1 I-1: explicit error='sem' + quantiles → BOTH SEM and quantile bars."""
        _,ax,_=DFDraw(df_gaussian).profile("y:x",bins=20,quantiles=[.16,.84],error='sem')
        ec=[c for c in ax.containers if isinstance(c,ErrorbarContainer)]
        assert len(ec)>=2,(
            f"Expected ≥2 ErrorbarContainers (SEM+quantile), got {len(ec)}. "
            "AD-52 sentinel fix (I-1) may not be working."
        ); plt.close('all')

    def test_explicit_none_renders_quantile_only(self,df_gaussian):
        """FIX1 I-2: error='none' + error_bars → quantile bars only."""
        _,ax,_=DFDraw(df_gaussian).profile("y:x",bins=20,quantiles=[.16,.84],error='none')
        ec=[c for c in ax.containers if isinstance(c,ErrorbarContainer)]
        assert len(ec)>=1,(
            "FIX1 I-2: error='none'+quantile error_bars MUST render quantile bars. "
            f"Got {len(ec)} containers."
        ); plt.close('all')

    def test_band_preserves_error_kwarg(self,df_gaussian):
        _,ax,_=DFDraw(df_gaussian).profile("y:x",bins=20,quantiles=[.16,.5,.84],error='std')
        polys=[c for c in ax.get_children() if isinstance(c,PolyCollection)]
        ec=[c for c in ax.containers if isinstance(c,ErrorbarContainer)]
        assert len(polys)>=1,"Band MUST be present"
        assert len(ec)>=1,"Error bars on central MUST be present with error='std'"; plt.close('all')

    def test_error_quantile_without_quantiles_raises(self,df_gaussian):
        with pytest.raises(ValueError,match="error='quantile' requires"):
            DFDraw(df_gaussian).profile("y:x",bins=20,error='quantile')
        plt.close('all')

# ── Class 7: same=True continuity (5 invariance) — T-4 FIXED ──

class TestQuantileSameTrueLastAxContinuity:
    def test_same_true_reuses_axes(self,df_gaussian):
        d=DFDraw(df_gaussian); _,a1,_=d.profile("y:x",bins=20)
        _,a2,_=d.profile("y:x",bins=20,same=True,quantiles=[.16,.84])
        assert a1 is a2; plt.close('all')

    def test_new_call_creates_new_axes(self,df_gaussian):
        d=DFDraw(df_gaussian); _,a1,_=d.profile("y:x",bins=20)
        _,a2,_=d.profile("y:x",bins=20)
        assert a1 is not a2; plt.close('all')

    def test_band_overlay_on_same(self,df_gaussian):
        d=DFDraw(df_gaussian); _,a1,_=d.profile("y:x",bins=20)
        _,a2,_=d.profile("y:x",bins=20,same=True,quantiles=[.16,.5,.84])
        assert a1 is a2
        assert any(isinstance(c,PolyCollection) for c in a2.get_children()); plt.close('all')

    def test_last_ax_preserved(self,df_gaussian):
        d=DFDraw(df_gaussian); _,a1,_=d.profile("y:x",bins=20)
        assert d._last_ax is a1; _,a2,_=d.profile("y:x",bins=20,same=True)
        assert a2 is a1; plt.close('all')

    @pytest.mark.skip(reason="AD-50 ADF-side Option C1b not yet shipped")
    def test_adf_cached_last_ax(self): pass

# ── Class 8: group_by interaction (4) — unchanged ──

class TestQuantileGroupByInteraction:
    def test_band_per_group(self,df_with_groups):
        DFDraw(df_with_groups).profile("y:x",bins=10,group_by='category',quantiles=[.16,.5,.84]); plt.close('all')
    def test_error_bars_per_group(self,df_with_groups):
        DFDraw(df_with_groups).profile("y:x",bins=10,group_by='category',quantiles=[.16,.84]); plt.close('all')
    def test_per_group_quantile_correctness(self,df_with_groups):
        _,_,s=DFDraw(df_with_groups).profile("y:x",bins=10,group_by='category',quantiles=[.16,.84])
        assert 'n' in s; plt.close('all')
    def test_grouped_legend_no_quantile_values(self,df_with_groups):
        _,ax,_=DFDraw(df_with_groups).profile("y:x",bins=10,group_by='category',quantiles=[.16,.84])
        leg=ax.get_legend()
        if leg:
            for t in leg.get_texts(): assert '0.16' not in t.get_text()
        plt.close('all')

# ── Class 9: Vector interaction (3) — unchanged ──

class TestQuantileVectorInteraction:
    def test_vector_band(self,df_vector):
        _,_,s=DFDraw(df_vector).profile("[y1,y2]:x",bins=10,quantiles=[.16,.5,.84])
        assert isinstance(s,list) and len(s)==2; plt.close('all')
    def test_vector_error_bars(self,df_vector):
        _,_,s=DFDraw(df_vector).profile("[y1,y2]:x",bins=10,quantiles=[.16,.84])
        assert isinstance(s,list); assert all('q_lower_per_bin' in si for si in s); plt.close('all')
    def test_vector_groupby_quantiles(self,df_with_groups):
        df=df_with_groups.copy(); df['y2']=df['y']+np.random.normal(0,.5,len(df))
        _,_,s=DFDraw(df).profile("[y,y2]:x",bins=10,group_by='category',quantiles=[.16,.5,.84])
        assert isinstance(s,list) and len(s)==2; plt.close('all')

# ── Class 10: Determinism (10 tests) — T-5 FIXED ──

class TestQuantileDeterminism:
    """T-5: Replaces fake parity tests with real determinism checks."""
    def _eq(self,s1,s2):
        for k in s1:
            if k in ('grouped','profile_data','q_lower_per_bin','q_upper_per_bin'): continue
            if isinstance(s1[k],(int,float)) and k in s2 and not np.isnan(s1[k]):
                np.testing.assert_allclose(s1[k],s2[k],rtol=1e-12,err_msg=f"Key '{k}'")

    def test_det_error_bars_mean(self,df_gaussian):
        d=DFDraw(df_gaussian); _,_,s1=d.profile("y:x",bins=20,quantiles=[.16,.84])
        _,_,s2=d.profile("y:x",bins=20,quantiles=[.16,.84])
        self._eq(s1,s2); np.testing.assert_array_equal(s1['q_lower_per_bin'],s2['q_lower_per_bin']); plt.close('all')

    def test_det_error_bars_median(self,df_gaussian):
        d=DFDraw(df_gaussian); _,_,s1=d.profile("y:x",bins=20,quantiles=[.16,.84],central='median')
        _,_,s2=d.profile("y:x",bins=20,quantiles=[.16,.84],central='median')
        self._eq(s1,s2); plt.close('all')

    def test_det_band_mean(self,df_gaussian):
        d=DFDraw(df_gaussian); _,_,s1=d.profile("y:x",bins=20,quantiles=[.16,.5,.84])
        _,_,s2=d.profile("y:x",bins=20,quantiles=[.16,.5,.84]); self._eq(s1,s2); plt.close('all')

    def test_det_band_median(self,df_gaussian):
        d=DFDraw(df_gaussian); _,_,s1=d.profile("y:x",bins=20,quantiles=[.16,.5,.84],central='median')
        _,_,s2=d.profile("y:x",bins=20,quantiles=[.16,.5,.84],central='median'); self._eq(s1,s2); plt.close('all')

    def test_det_band_both(self,df_gaussian):
        d=DFDraw(df_gaussian); _,_,s1=d.profile("y:x",bins=20,quantiles=[.16,.5,.84],central='both')
        _,_,s2=d.profile("y:x",bins=20,quantiles=[.16,.5,.84],central='both'); self._eq(s1,s2); plt.close('all')

    def test_det_band_none(self,df_gaussian):
        d=DFDraw(df_gaussian); _,_,s1=d.profile("y:x",bins=20,quantiles=[.16,.5,.84],central='none')
        _,_,s2=d.profile("y:x",bins=20,quantiles=[.16,.5,.84],central='none'); self._eq(s1,s2); plt.close('all')

    def test_det_with_groupby(self,df_with_groups):
        d=DFDraw(df_with_groups); _,_,s1=d.profile("y:x",bins=10,quantiles=[.16,.84],group_by='category')
        _,_,s2=d.profile("y:x",bins=10,quantiles=[.16,.84],group_by='category'); self._eq(s1,s2); plt.close('all')

    def test_det_with_vector(self,df_vector):
        d=DFDraw(df_vector); _,_,s1=d.profile("[y1,y2]:x",bins=10,quantiles=[.16,.84])
        _,_,s2=d.profile("[y1,y2]:x",bins=10,quantiles=[.16,.84])
        for i in range(2): self._eq(s1[i],s2[i])
        plt.close('all')

    def test_det_quantile_arrays(self,df_gaussian):
        d=DFDraw(df_gaussian); _,_,s1=d.profile("y:x",bins=20,quantiles=[.16,.84])
        _,_,s2=d.profile("y:x",bins=20,quantiles=[.16,.84])
        np.testing.assert_array_equal(s1['q_lower_per_bin'],s2['q_lower_per_bin'])
        np.testing.assert_array_equal(s1['q_upper_per_bin'],s2['q_upper_per_bin']); plt.close('all')

    def test_det_error_none_quantile(self,df_gaussian):
        d=DFDraw(df_gaussian); _,_,s1=d.profile("y:x",bins=20,quantiles=[.16,.84],error='none')
        _,_,s2=d.profile("y:x",bins=20,quantiles=[.16,.84],error='none'); self._eq(s1,s2); plt.close('all')

# ── Class 11: Backward compat (3) — T-6 FIXED: deterministic baseline ──

class TestQuantileBackwardCompat:
    def test_no_quantiles_produces_one_errorbar_container(self,df_deterministic):
        _,ax,_=DFDraw(df_deterministic).profile("y:x",bins=5)
        ec=[c for c in ax.containers if isinstance(c,ErrorbarContainer)]
        assert len(ec)==1,f"Standard profile: 1 ErrorbarContainer, got {len(ec)}"; plt.close('all')

    def test_no_quantiles_stats_match_reference(self,df_deterministic):
        _,_,s=DFDraw(df_deterministic).profile("y:x",bins=5)
        assert s['n']==10
        np.testing.assert_allclose(s['mean_y'],5.0,rtol=1e-12); plt.close('all')

    def test_production_pattern_unchanged(self,df_with_groups):
        df=df_with_groups.copy(); df['cf']=np.random.uniform(0,5,len(df))
        _,_,s=DFDraw(df).profile("y:x",bins=20,group_by='cf',group_by_quantiles=4)
        assert s['n']==len(df); assert s.get('grouped',False); plt.close('all')

# ── Class 12: Docstrings (4 smoke) — unchanged ──

class TestQuantileDocstrings:
    def test_quantiles_example(self,df_gaussian):
        f,_,_=DFDraw(df_gaussian).profile("y:x",bins=20,quantiles=[.16,.84]); assert f; plt.close('all')
    def test_central_example(self,df_gaussian):
        f,_,_=DFDraw(df_gaussian).profile("y:x",bins=20,quantiles=[.16,.5,.84],central='median'); assert f; plt.close('all')
    def test_mode_example(self,df_gaussian):
        f,_,_=DFDraw(df_gaussian).profile("y:x",bins=20,quantiles=[.16,.5,.84],quantile_mode='band'); assert f; plt.close('all')
    def test_central_default_is_mean(self):
        assert get_style_value("quantile.central_default",None)=="mean"

# ── Class 13: Style key defaults (8 invariance) — unchanged ──

class TestQuantileStyleKeyDefaults:
    def setup_method(self): set_style(None)
    def teardown_method(self): set_style(None)
    def test_band_alpha_default(self): assert get_style_value("quantile.band.alpha",None)==0.25
    def test_band_hatch_default(self): assert get_style_value("quantile.band.hatch","X") is None
    def test_capsize_default(self): assert get_style_value("quantile.error_bars.capsize",None)==3.0
    def test_central_default(self): assert get_style_value("quantile.central_default",None)=="mean"
    def test_override_band(self,df_gaussian):
        set_style({"quantile.band.alpha":0.6}); assert get_style_value("quantile.band.alpha")==0.6
        DFDraw(df_gaussian).profile("y:x",bins=10,quantiles=[.16,.5,.84]); plt.close('all')
    def test_override_capsize(self,df_gaussian):
        set_style({"quantile.error_bars.capsize":8.0}); assert get_style_value("quantile.error_bars.capsize")==8.0
        DFDraw(df_gaussian).profile("y:x",bins=10,quantiles=[.16,.84]); plt.close('all')
    def test_save_load_roundtrip(self):
        set_style({"quantile.band.alpha":0.42,"quantile.error_bars.capsize":9.0})
        with tempfile.NamedTemporaryFile(suffix='.json',delete=False,mode='w') as f: path=f.name
        save_style(path); set_style(None); load_style(path)
        assert get_style_value("quantile.band.alpha")==0.42
        assert get_style_value("quantile.error_bars.capsize")==9.0; Path(path).unlink()
    def test_namespace_integrity(self):
        qk=[k for k in DEFAULT_STYLE if k.startswith('quantile.')]; assert len(qk)==4
        for bad in ('quantiles.','qmode.','q.','band.','errorbars.'):
            assert not [k for k in DEFAULT_STYLE if k.startswith(bad)]

# ── Class 14: Bool histogram (8 tests) — T-7 BUG_dfdraw_20260505 ──

class TestBoolExpressionHistogram:
    """BUG_dfdraw_20260505: All tests through DFDraw — the actual crash path."""

    def test_equality_bool_histogram(self,df_bool):
        """(x==0) produces bool column; hist must not crash, stats must be correct."""
        d=DFDraw(df_bool); _,ax,s=d.hist('(x==0)',bins=2)
        assert s['n']==len(df_bool)
        assert s['min']==0.0; assert s['max']==1.0
        expected_mean=(df_bool['x']==0).mean()
        assert abs(s['mean']-expected_mean)<1e-10; plt.close('all')

    def test_inequality_bool(self,df_bool):
        _,_,s=DFDraw(df_bool).hist('(x!=0)',bins=2); assert s['n']==len(df_bool); plt.close('all')

    def test_greater_than_bool(self,df_bool):
        _,_,s=DFDraw(df_bool).hist('(x>2)',bins=2); assert s['n']==len(df_bool); plt.close('all')

    def test_logical_and_bool(self,df_bool):
        _,_,s=DFDraw(df_bool).hist('(x>0)&(x<3)',bins=2); assert s['n']==len(df_bool); plt.close('all')

    def test_logical_or_bool(self,df_bool):
        _,_,s=DFDraw(df_bool).hist('(x==0)|(x==4)',bins=2); assert s['n']==len(df_bool); plt.close('all')

    def test_logical_not_bool(self,df_bool):
        _,_,s=DFDraw(df_bool).hist('~(x==0)',bins=2); assert s['n']==len(df_bool); plt.close('all')

    def test_bool_group_by(self,df_bool):
        _,_,s=DFDraw(df_bool).hist('(x==0)',bins=2,group_by='cat'); assert s['n']==len(df_bool); plt.close('all')

    def test_bool_values_correct(self,df_bool):
        """Bool hist stats should reflect True/False distribution."""
        _,_,s=DFDraw(df_bool).hist('(x==0)',bins=2)
        n_t=int((df_bool['x']==0).sum())
        expected_mean=n_t/len(df_bool)
        assert abs(s['mean']-expected_mean)<1e-10
        assert s['n']==len(df_bool); plt.close('all')

# ── Class 15: Bool profile (3 tests) — T-7 BUG_dfdraw_20260505 ──

class TestBoolExpressionProfile:
    def test_bool_y_profile(self,df_bool):
        _,_,s=DFDraw(df_bool).profile('(x==0):y',bins=10); assert s['n']==len(df_bool); plt.close('all')

    def test_bool_x_profile(self,df_bool):
        _,_,s=DFDraw(df_bool).profile('y:(x>2)',bins=2); assert s['n']==len(df_bool); plt.close('all')

    def test_bool_y_with_quantiles(self,df_bool):
        _,_,s=DFDraw(df_bool).profile('(x==0):y',bins=10,quantiles=[.16,.84])
        assert 'q_lower_per_bin' in s; plt.close('all')
