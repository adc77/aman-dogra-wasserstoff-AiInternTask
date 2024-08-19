import json
import os
import uuid

class DataMapper:
    def __init__(self, output_dir='data/mapped_data'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.master_data = {}

    def map_data(self, input_image_path, segmentation_data, analysis_data):
        master_id = str(uuid.uuid4())
        self.master_data[master_id] = {}
        
        for obj in segmentation_data:
            object_id = obj['id']
            self.master_data[master_id][object_id] = {
                'bbox': obj.get('bbox', []),
                'segmentation': obj.get('segmentation', []),
                'identification': analysis_data.get(object_id, {}).get('identification', ''),
                'extracted_text': analysis_data.get(object_id, {}).get('extracted_text', ''),
                'summary': analysis_data.get(object_id, {}).get('summary', '')
            }
        
        return master_id

    def save_mapping(self, filename='mapped_data.json'):
        with open(os.path.join(self.output_dir, filename), 'w') as f:
            json.dump(self.master_data, f, indent=2)

    def load_mapping(self, filename='mapped_data.json'):
        filepath = os.path.join(self.output_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                self.master_data = json.load(f)
        return self.master_data

class OutputGenerator:
    def __init__(self, output_dir='data/output'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_output(self, master_id, mapped_data, original_image):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import pandas as pd

        # Create the figure and axes
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        # Display the original image with bounding boxes
        ax1.imshow(original_image)
        ax1.set_title('Original Image with Detected Objects')

        # Prepare data for the table
        table_data = []

        for obj_id, obj_info in mapped_data[master_id].items():
            # Draw bounding box
            bbox = obj_info['bbox']
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                                     linewidth=2, edgecolor='r', facecolor='none')
            ax1.add_patch(rect)
            ax1.text(bbox[0], bbox[1], obj_id, color='r')

            # Prepare table data
            table_data.append({
                'Object ID': obj_id,
                'Identification': obj_info['identification'],
                'Extracted Text': obj_info['extracted_text'],
                'Summary': obj_info['summary']
            })

        # Create the table
        df = pd.DataFrame(table_data)
        ax2.axis('off')
        ax2.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
        ax2.set_title('Object Data Summary')

        # Adjust layout and save the figure
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, f'{master_id}_output.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Save table data as CSV
        csv_path = os.path.join(self.output_dir, f'{master_id}_table.csv')
        df.to_csv(csv_path, index=False)

        return output_path, csv_path