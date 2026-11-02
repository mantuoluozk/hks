"""Run benchmark and optional retrieval evaluation with one native model load."""
import os
from pathlib import Path
import sys
import native_runtime


def main():
    out=Path(os.environ['RUN_RESULTS_DIR'])
    llm=native_runtime.load_model()
    native_runtime.load_model=lambda:llm
    import benchmark_native
    os.environ['RUN_RESULTS_DIR']=str(out/'performance')
    benchmark_native.main()
    if os.environ.get('SKIP_CHARTQA')!='1':
        import evaluate_chartqa
        os.environ['RUN_RESULTS_DIR']=str(out/'chartqa')
        sys.argv=[sys.argv[0],'--backend','native']
        evaluate_chartqa.main()


if __name__=='__main__':main()
