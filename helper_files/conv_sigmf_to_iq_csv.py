"""
This file is here to convert sigmf-data files into csv files containing 
IQ samples. 
We belliieeeeve that each line relates to a single sample. 
"""
    
import argparse
import json
import numpy as np


ap = argparse.ArgumentParser(description="Windowed SigMF energy summary")
ap.add_argument("--meta", help="Path to .sigmf-meta file")
ap.add_argument("-o", "--out", default="iqSamples.csv", help="Output text file")
args = ap.parse_args()

def runConversion(): 
    meta_file = args.meta
    data_file = args.meta.replace("meta", "data")
    # text_output = args.meta.replace("sigmf-meta", "csv")
    text_output = args.out

    print("--------------------------------------------")
    print(f"Working on: \n{meta_file}")

    try:
        with open(meta_file, 'r') as f:
            meta = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: {json.JSONDecodeError}")
        return

    dtype_map = {
        "ri8_le":  np.int8,
        "ri16_le": np.int16,
        "ri32_le": np.int32,
        "rf32_le": np.float32,
        "cf32_le": np.complex64,
        "ci8_le":  np.int8,
        "ci16_le": np.int16,
        "ci32_le": np.int32,
    }
    dtype = dtype_map.get(meta["global"]["core:datatype"], np.int16)

    iq = np.fromfile(data_file, dtype=dtype)
    iq_complex = iq[::2] + 1j * iq[1::2]

    with open(text_output, 'w') as f:
        try:
            f.write(f"i,q\n")
        except Exception as e:
            print(f"Error writing header ('i,q'): {e}")
        for sample in iq_complex:
            try:
                f.write(f"{sample.real},{sample.imag}\n")
            except Exception as e:
                print(f"Error converting: {e}")

    print(f" Saved {len(iq_complex)} samples to {text_output}")


def main(): 
    if "sigmf-meta" not in args.meta: 
        print(f"This is not a meta file. Exiting.")
    else: 
        runConversion()


main()