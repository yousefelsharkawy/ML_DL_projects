## Imports 
import re
import pandas as pd
import argparse

def interpolate(input_file,output_file):
    ## read the csv file
    df = pd.read_csv(input_file)


    ## prepare the data for interpolation

    # replace the ' ' with ',' in the bbox column
    df['car_bbox'] = df['car_bbox'].apply(lambda x: re.sub(' ', ',', x))
    df['license_plate_bbox'] = df['license_plate_bbox'].apply(lambda x: re.sub(' ', ',', x))

    # convert the list of string representation of lists to actual lists
    df['car_bbox'] = df['car_bbox'].apply(lambda x: eval(x))
    df['license_plate_bbox'] = df['license_plate_bbox'].apply(lambda x: eval(x))
    # expand the bbox column to 4 columns
    df[['car_x1', 'car_y1', 'car_x2', 'car_y2']] = pd.DataFrame(df['car_bbox'].tolist(), index=df.index)
    df[['license_plate_x1', 'license_plate_y1', 'license_plate_x2', 'license_plate_y2']] = pd.DataFrame(df['license_plate_bbox'].tolist(), index=df.index)
    # drop the bbox column
    df = df.drop(columns=['car_bbox', 'license_plate_bbox'])

    ## interpolate the data
    # initialize a list to store the interpolated data
    interpolated_data = []

    for car_id in df.car_id.unique():
        car_rows = df[df.car_id == car_id].set_index('frame_nmr').sort_index()
        # create a complete sequence of frames from min frame to max frame
        all_frames = pd.DataFrame(index=range(car_rows.index.min(), car_rows.index.max() + 1))
        car_rows_complete = car_rows.reindex(all_frames.index)
        # fill the missing values for the car_id, license_number and license_number_confidence
        car_rows_complete['car_id'] = car_id
        car_rows_complete['license_number'] = car_rows_complete['license_number'].ffill().bfill()
        car_rows_complete['license_number_confidence'] = car_rows_complete['license_number_confidence'].ffill().bfill()
        ## interpolate the car_bbox_confidence and license_plate_bbox_confidence
        car_rows_complete['car_bbox_confidence'] = car_rows_complete['car_bbox_confidence'].interpolate(method='nearest')
        car_rows_complete['license_plate_bbox_confidence'] = car_rows_complete['license_plate_bbox_confidence'].interpolate(method='nearest')
        # interpolate the car_bbox and license_plate_bbox
        for col in ['car_x1', 'car_y1', 'car_x2', 'car_y2', 'license_plate_x1', 'license_plate_y1', 'license_plate_x2', 'license_plate_y2']:
            car_rows_complete[col] = car_rows_complete[col].interpolate(method='linear')
        # rejoin the coordinates into a single bbox column list
        car_rows_complete['car_bbox'] = car_rows_complete[['car_x1', 'car_y1', 'car_x2', 'car_y2']].values.tolist()
        car_rows_complete['license_plate_bbox'] = car_rows_complete[['license_plate_x1', 'license_plate_y1', 'license_plate_x2', 'license_plate_y2']].values.tolist()
        # drop the individual coordinates columns
        car_rows_complete = car_rows_complete.drop(columns=['car_x1', 'car_y1', 'car_x2', 'car_y2', 'license_plate_x1', 'license_plate_y1', 'license_plate_x2', 'license_plate_y2'])
        
        # append the interpolated data to the list
        interpolated_data.append(car_rows_complete.reset_index().rename(columns={'index': 'frame_nmr'}))    

    # concatenate the interpolated data
    df_interpolated = pd.concat(interpolated_data).sort_values(by=['frame_nmr', 'car_id']).reset_index(drop=True)
    # drop the license_number_confidence column
    df_interpolated = df_interpolated.drop(columns=['license_number_confidence'])

    ## save the result
    print("Interpolation completed successfully!")
    df_interpolated.to_csv(output_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interpolate the results of the license plate reader.")
    parser.add_argument("--input_file", type=str, required=True, help="Input CSV file with the results of inference.py file.")
    parser.add_argument("--output_file", type=str,  required=True, help="Output CSV file to save the interpolated results.")
    args = parser.parse_args()

    interpolate(args.input_file, args.output_file)