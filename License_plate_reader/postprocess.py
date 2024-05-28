## imports 
import pandas as pd
import argparse



def postprocess(input_file,output_file,keep_medium):
    ## read the csv file
    df = pd.read_csv(input_file)
    
    ## Unify the license numbers according to the confidence
    unique_ids = df.car_id.unique()
    for car_id in unique_ids:
        # extract the rows for the car_id
        car_rows = df[df.car_id == car_id]
        # check if there is a high confidence license number, and get the most common license number read by the OCR
        if 'high' in car_rows.license_number_confidence.values:
            # get the most common license number
            license_number = car_rows[car_rows['license_number_confidence'] == 'high']['license_number'].mode()[0]
            # update the license number for all the rows of this car_id
            df.loc[df.car_id == car_id, 'license_number'] = license_number
            # update the confidence for all the rows of this car_id
            df.loc[df.car_id == car_id, 'license_number_confidence'] = 'high'
        # if there is no high confidence, check if there is a medium confidence, and get the most common license number read by the OCR in it
        elif 'medium' in car_rows.license_number_confidence.values:
            if keep_medium:
                license_number = car_rows[car_rows['license_number_confidence'] == 'medium']['license_number'].mode()[0]
                df.loc[df.car_id == car_id, 'license_number'] = license_number
                df.loc[df.car_id == car_id, 'license_number_confidence'] = 'medium'
            else:
                df = df[df.car_id != car_id]
        # discard the rows with only low confidence
        else:
            # if there is no high or medium confidence, delete the rows for this car_id
            df = df[df.car_id != car_id]
    
    
    ## Unify the car_ids according to the license number
    # sometimes the tracker may assign different car_ids to the same car, so we will unify them if they have the same license number
    unique_license_numbers = df.license_number.unique()
    for license_number in unique_license_numbers:
        # get all the rows for the license_number
        license_rows = df[df.license_number == license_number]
        # if there are multiple car_ids for the same license number, get the most common car_id and update all the rows
        if license_rows.car_id.nunique() > 1:
            car_id = license_rows['car_id'].mode()[0]
            df.loc[df.license_number == license_number, 'car_id'] = car_id
    
    print("Precprocessing completed successfully!")
    ## Write the results to a new csv file
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Postprocess the results of the license plate reader.")
    parser.add_argument("--input_file", type=str, required=True, help="Input CSV file with the results of inference.py file.")
    parser.add_argument("--output_file", type=str,  required=True, help="Output CSV file to save the postprocessed results.")
    parser.add_argument("--keep_medium", default=False, action='store_true', help="Keep the medium confidence license numbers.")
    args = parser.parse_args()

    postprocess(args.input_file, args.output_file, args.keep_medium)