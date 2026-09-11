import matplotlib; matplotlib.use('Agg')
import sys, warnings, pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from dfextensions.dfdraw import DFDraw

OUT = sys.argv[1] if len(sys.argv) > 1 else "summary_fit_semantics.pdf"
BR, REP, FV, XC, EPS = (0,1,2), {0:5,1:7,2:11}, (10,20), (0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5), 1e-3

rows=[]
for b in BR:
    k=REP[b]
    for fi,f in enumerate(FV):
        s=10.0*(b+1)+(fi+1); i0=100.0*b+3.0*fi
        for x in XC:
            for j in range(k):
                rows.append({"branch":b,"facet_val":f,"x":x,"y":i0+s*x+(j-(k-1)/2.0)*EPS})
df=pd.DataFrame(rows)

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    fig,axes,st = DFDraw(df).profile("y:x", bins=8, range=(0.0,8.0),
        selection_vector=[f"branch=={b}" for b in BR],
        selection_labels=[f"B{b}" for b in BR],
        facet_by="facet_val", vector_compose="outer",
        fit="pol1", summary_fit="table", return_data=True, auto_title=True)

payload = st[0]["summary_fit"]
with PdfPages(OUT) as pdf:
    fig.suptitle("vector x facet profile: 3 branches x 2 facets", fontsize=10)
    pdf.savefig(fig)
    pdf.savefig(payload["table"])
print(f"wrote {OUT}")
print(f"rows: {len(payload['data'])}  (expect 3 branches x 2 facets = 6)")
print(f"{'facet label':<26} {'group':<7} {'slope':>8} {'expect':>8} {'intercept':>10} {'expect':>8}")
for r in payload["data"]:
    lab=str(r.get("facet")); f=int(lab.split("facet_val=")[1].split()[0]); b=int(lab.split("branch=")[1])
    fi=FV.index(f)
    print(f"{lab:<26} {str(r.get('group')):<7} {r.get('slope'):>8.3f} "
          f"{10.0*(b+1)+(fi+1):>8.1f} {r.get('intercept'):>10.3f} {100.0*b+3.0*fi:>8.1f}")
