from data.dataset_generator import DatasetGenerator

def main():
    dataset_generator = DatasetGenerator(data_count=800)
    df = dataset_generator.generate_dataset()



