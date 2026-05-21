using MLBenchmarks
import MLBenchmarks: mse, rmse
import MLBenchmarks: run_experiment

using CSV
using Zygote

data_name = :year
modernnca_hyper_size = 8
data = load_data(data_name; uniformize=true)
metrics = [:mse, :rmse]
mkpath(joinpath("results", string(data_name)))

################################
# ModernNCA
################################
hyper_list = MLBenchmarks.get_hyper_modernnca(modernnca_hyper_size; data.loss, data.metric, device=:gpu, backend=:zygote, nrounds=200, early_stopping_rounds=2, lr=1e-3, d_embedding=[64, 128], d_block=[128, 256], n_blocks=[1, 2], dropout=0.1, temperature=1.0, sample_rate=0.01, batchsize=1024, seed=123)
results_df = run_experiment(:NeuroTabModels, data, hyper_list; metrics)
CSV.write(joinpath("results", string(data_name), "modernnca.csv"), results_df)

hyper_list = MLBenchmarks.get_hyper_talent_modernnca(modernnca_hyper_size; nrounds=200, lr=1e-3, dim=[64, 128], d_block=[128, 256], n_blocks=[1, 2], dropout=0.1, temperature=1.0, sample_rate=0.01, batchsize=1024, seed=123, normalization="none")
results_df = run_experiment(:TALENT, data, hyper_list; metrics, save_root=joinpath("results", string(data_name), "talent_checkpoints"))
CSV.write(joinpath("results", string(data_name), "talent_modernnca.csv"), results_df)
