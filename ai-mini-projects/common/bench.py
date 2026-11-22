from time import perf_counter
import statistics
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from datetime import datetime

class Bench():
    def bench_load(self, md):
        self.md = md
        start = perf_counter()
        md.load()
        elapse = perf_counter() - start

        print(f"Load model in {elapse}")
        return elapse

    def bench_predict(self, request):
        start = perf_counter()
        result = self.md.predict(request)
        elapse = perf_counter() - start
        return {
            "elapse": elapse,
            "metrics": result["metrics"]
        }

    def load_and_bench(self, md, requests, concurrency: int = 1):
        load_elapse = self.bench_load(md)
        print(f"Bench for {len(requests)}")

        result = {
            "load_elapse": load_elapse,
            "latencies": [],
            "metrics": []
        }
        start_tp = perf_counter()

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            for rs in executor.map(self.bench_predict, requests):
                result["latencies"].append(rs["elapse"])
                result["metrics"].append(rs["metrics"])

        
        q =  statistics.quantiles(result["latencies"], n=100)
        result["p50"] = q[49]
        result["p95"] = q[94]
        result["p99"] = q[98]

        result["throughput"] = len(requests) / (perf_counter() - start_tp)
        return result

    def save(self, result, config):
        Path("results").mkdir(exist_ok=True)
        name = datetime.now().strftime("%Y%m%d-%H%M%S-%f")

        data_to_save = {
            "config": config,
            "result": result
        }
        with open(f"results/{name}.json", "w") as f:
            json.dump(data_to_save, f, indent=2)



if __name__ == "__main__":
    from dummy import Dummy
    model = Dummy(0.1)
    concurrency = 1
    requests = ["a"] * 20
    r = Bench().load_and_bench(model, requests, concurrency=concurrency)

    Bench().save(result=r, config={
        "model": model.get_name(),
        "concurrency": concurrency,
        "requests": len(requests),
    })