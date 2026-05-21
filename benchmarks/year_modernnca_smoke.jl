using MLBenchmarks
import MLBenchmarks: rmse
import MLBenchmarks: run_experiment

using CSV
using DataFrames
using Zygote

data_name = :year
data = load_data(data_name; uniformize=true)
metrics = [:mse, :rmse]

function subset_rows(df, n)
    df[1:min(n, nrow(df)), :]
end

data = merge(data, (;
    dtrain=subset_rows(data.dtrain, 4096),
    deval=subset_rows(data.deval, 1024),
    dtest=subset_rows(data.dtest, 1024),
))

outdir = joinpath("results", string(data_name), "modernnca_smoke")
mkpath(outdir)

hyper_list = MLBenchmarks.get_hyper_modernnca(
    1;
    data.loss,
    data.metric,
    device=:cpu,
    backend=:zygote,
    nrounds=2,
    early_stopping_rounds=1,
    lr=1e-3,
    d_embedding=32,
    d_block=64,
    n_blocks=1,
    dropout=0.1,
    temperature=1.0,
    sample_rate=0.01,
    batchsize=256,
    seed=123,
)
results_df = run_experiment(:NeuroTabModels, data, hyper_list; metrics)
CSV.write(joinpath(outdir, "modernnca.csv"), results_df)

hyper_list = MLBenchmarks.get_hyper_talent_modernnca(
    1;
    nrounds=2,
    lr=1e-3,
    dim=32,
    d_block=64,
    n_blocks=1,
    dropout=0.1,
    temperature=1.0,
    sample_rate=0.01,
    batchsize=256,
    seed=123,
    normalization="none",
)
results_df = run_experiment(:TALENT, data, hyper_list; metrics, save_root=joinpath(outdir, "talent_checkpoints"))
CSV.write(joinpath(outdir, "talent_modernnca.csv"), results_df)
