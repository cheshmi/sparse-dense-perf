import json
import matplotlib.pyplot as plt
import os


def load_reports(directory):
    """Load all JSON reports from a given directory."""
    reports = []
    for file in os.listdir(directory):
        if file.endswith(".json"):
            with open(os.path.join(directory, file)) as f:
                reports.append(json.load(f))
                # add clock field to the reports from file name
                reports[-1]["clock"] = file.split("_")[1]
    return reports


def extract_data(reports):
    """Extract relevant data from the reports."""
    data = {}  # Dictionary to organize data
    for report in reports:
        for benchmark in report["benchmarks"]:
            name = benchmark["name"]
            real_time = benchmark.get("real_time", None)

            # Parse key details from the benchmark name
            # Assuming the naming convention is like: "BM_Method/K=2/Sparsity=50"
            if name.startswith("BM_"):
                try:
                    parts = name.split("/")
                    method = parts[0][3:]  # Extract method (e.g., cublas, cusparse)
                    k_value = int(parts[3].split(":")[1])  # Extract K value
                    sparsity = int(parts[4].split(":")[1])  # Extract sparsity value
                    clock_value = report["clock"]

                    # Organize data by K, then by method and clock
                    # Organize data by K, then by clock and method
                    if k_value not in data:
                        data[k_value] = {}
                    if clock_value not in data[k_value]:
                        data[k_value][clock_value] = {"cuBLAS_CUDA": {}, "CUSPARSE_SPMM": {}}

                    # Store real_time data for the specific method and sparsity
                    data[k_value][clock_value][method][sparsity] = real_time
                    # if method is cublas, store the time for all sparsities 60 70 80 90 95 99
                    if method == "cuBLAS_CUDA":
                        data[k_value][clock_value][method][60] = real_time
                        data[k_value][clock_value][method][70] = real_time
                        data[k_value][clock_value][method][80] = real_time
                        data[k_value][clock_value][method][90] = real_time
                        data[k_value][clock_value][method][95] = real_time
                        data[k_value][clock_value][method][99] = real_time



                except Exception as e:
                    print(f"Skipping invalid benchmark entry: {name} - {e}")
    return data



def calculate_ratios(data):
    """Calculate the ratio of cublas_time to cusparse_time for each K and clock."""
    ratios = {}

    for k_value, clocks in data.items():
        ratios[k_value] = {}
        for clock_value, methods in clocks.items():
            ratios[k_value][clock_value] = []

            # Get sparsity values common to both cublas and cusparse
            cublas_data = methods["cuBLAS_CUDA"]
            cusparse_data = methods["CUSPARSE_SPMM"]
            common_sparsity = set(cublas_data.keys()).intersection(cusparse_data.keys())

            # Calculate ratios (cublas_time / cusparse_time) for common sparsity values
            for sparsity in sorted(common_sparsity):
                cublas_time = cublas_data[sparsity]
                cusparse_time = cusparse_data[sparsity]
                ratio = cublas_time / cusparse_time
                ratios[k_value][clock_value].append((sparsity, ratio))
    return ratios


def plot_results(ratios):
    """Generate plots for the extracted data."""
    for k_value, clocks in ratios.items():
        plt.figure(figsize=(10, 6))

        # Plot each clock configuration as a separate line
        for clock_value, values in clocks.items():
            if values:
                # Sort by sparsity for clean plotting
                sorted_data = sorted(values, key=lambda x: x[0])  # Sort by sparsity
                sparsity, ratio = zip(*sorted_data)

                # Plot the ratio
                plt.plot(
                    sparsity,
                    ratio,
                    marker="o",
                    label=f"Clock={clock_value}"
                )

        # Customize the plot
        plt.title(f"Ratio of cublas_time / cusparse_time for K={k_value}")
        plt.xlabel("Sparsity (%)")
        plt.ylabel("Ratio (cublas_time / cusparse_time)")
        # set y-label to be between 0 to 5.5
        plt.ylim(0, 5.5)
        # make a straitg line in y=1
        plt.axhline(y=1, color='r', linestyle='--')
        plt.legend()
        plt.grid(True)

        # Save the plot to a file
        plt.savefig(f"plot_K={k_value}_ratio.png")
        #plt.show()
        plt.close()




def main():
    # Directory containing the JSON reports
    json_directory = "./report"  # Update this to the directory containing the JSON files

    # Step 1: Load all reports
    reports = load_reports(json_directory)
    if not reports:
        print("No JSON reports found in the specified directory.")
        return

    # Step 2: Extract benchmark data
    data = extract_data(reports)

    # Step 3: Calculate the cublas_time / cusparse_time ratios
    ratios = calculate_ratios(data)

    # Step 4: Plot the results
    plot_results(ratios)



if __name__ == "__main__":
    main()
