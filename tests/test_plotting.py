from run_benchmarks import read_log_entries, plot_logs


def test_plot_logfile(benchmark_log_file):
    log_entries = read_log_entries(benchmark_log_file)
    chart = plot_logs(log_entries, use_time_index=True)
    chart.show()



