using MLBenchmarks
import MLBenchmarks: mse, rmse

using CSV
using DataFrames
using NeuroTabModels
using Reactant

data_name = :year
data = load_data(data_name; uniformize=false)
outdir = joinpath("results", string(data_name))
mkpath(outdir)

################################
# ModernNCA settings aligned with NeuroTabModels.jl benchmark_mse/YEAR regression
################################
arch = NeuroTabModels.ModernNCAConfig(;
    d_embedding=32,
    n_blocks=1,
    d_block=64,
    dropout=0.1f0,
    sample_rate=0.1,
)

embedding_config = NeuroTabModels.EmbeddingLayer(
    num=NeuroTabModels.BatchNormEmbeddings(),
)

learner = NeuroTabModels.NeuroTabRegressor(
    arch;
    embedding_config,
    loss=:mse,
    nrounds=20,
    early_stopping_rounds=2,
    lr=1e-3,
    batchsize=1024,
    device=:cpu,
    backend=:reactant,
)

train_time = @elapsed m = NeuroTabModels.fit(
    learner,
    data.dtrain;
    deval=data.deval,
    target_name=data.target_name,
    feature_names=data.feature_names,
    print_every_n=5,
)

p_eval = m(data.deval; device=:cpu)
p_eval = p_eval isa AbstractMatrix && size(p_eval, 2) == 1 ? p_eval[:, 1] : p_eval
eval_mse = mse(p_eval, data.deval[!, data.target_name])
eval_rmse = rmse(p_eval, data.deval[!, data.target_name])
@info "MSE/RMSE - deval" eval_mse eval_rmse

p_test = m(data.dtest; device=:cpu)
p_test = p_test isa AbstractMatrix && size(p_test, 2) == 1 ? p_test[:, 1] : p_test
test_mse = mse(p_test, data.dtest[!, data.target_name])
test_rmse = rmse(p_test, data.dtest[!, data.target_name])
@info "MSE/RMSE - dtest" test_mse test_rmse

results_df = DataFrame([(
    model_type="ModernNCA",
    backend="reactant",
    device="cpu",
    train_time=train_time,
    best_nround=m.info[:logger][:best_iter],
    batchsize=1024,
    epochs=20,
    d_embedding=32,
    n_blocks=1,
    d_block=64,
    dropout=0.1,
    sample_rate=0.1,
    embedding_config="BatchNormEmbeddings",
    eval_mse=eval_mse,
    test_mse=test_mse,
    eval_rmse=eval_rmse,
    test_rmse=test_rmse,
)])

CSV.write(joinpath(outdir, "modernnca_reactant.csv"), results_df)
