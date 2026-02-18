import json
import numpy as np
import argparse




# band_bin_cache = {}

# print("Spectrogram shape:", SPECTRUM.shape)
# print("FFT bin width:", Fs / NFFT, "Hz")






# def get_band_bins(rf_lo, rf_hi):
#     key = (rf_lo, rf_hi)
#     if key not in band_bin_cache:
#         band_bin_cache[key] = np.where(
#             (f_rf >= rf_lo) & (f_rf <= rf_hi)
#         )[0]
#     return band_bin_cache[key]





# def getPower(sample_start: int,
#              sample_count: int,
#              rf_lo: float,
#              rf_hi: float) -> float:
#     """
#     Returns mean integrated RF-band power in dBFS
#     using precomputed spectrogram.
#     """

#     # ---- Time → frame mapping ----
#     frame_start = sample_start // HOP
#     frame_end   = (sample_start + sample_count) // HOP

#     frame_start = max(frame_start, 0)
#     frame_end   = min(frame_end, SPECTRUM.shape[0])

#     if frame_end <= frame_start:
#         return float("-inf")

#     # ---- Frequency → bin mapping ----
#     # band_bins = np.where((f_rf >= rf_lo) & (f_rf <= rf_hi))[0]
#     band_bins = get_band_bins(rf_lo, rf_hi)

#     if band_bins.size == 0:
#         return float("-inf")

#     # ---- Integrate power ----
#     band_power = np.mean(
#         np.sum(SPECTRUM[frame_start:frame_end, band_bins], axis=1)
#     )

#     # ---- dBFS ----
#     return 10 * np.log10(band_power + 1e-20)




# def findMinPower(meta):
#     ''' 
#     findMinPower selects returns the power of the annotation
#     that the user selected, using [args.minann - 1]
#     '''
#     a = meta["annotations"][args.minann - 1]
#     rf_lo = a["core:freq_lower_edge"]
#     rf_hi = a["core:freq_upper_edge"]
#     sample_start = a["core:sample_start"]
#     sample_count = a["core:sample_count"]

#     minPower = getPower(sample_start, sample_count, rf_lo, rf_hi)
#     return round(minPower, 4)




# def loopAnnotations(metafile):
#     '''
#     loopAnnotations goes through a binary iq file and uses its
#     associated meta file in order to remove low power annotations.
#     Setup for the SigMF file format.
#     '''
#     with open(metafile) as f:
#         meta = json.load(f)
#     kept_data = meta

#     minimum_power_dbfs = findMinPower(meta) + 5
#     print(f"The min power: {minimum_power_dbfs} dBFS")

#     kept = []
#     for a in meta["annotations"]:
#         rf_lo = a["core:freq_lower_edge"]
#         rf_hi = a["core:freq_upper_edge"]
#         sample_start = a["core:sample_start"]
#         sample_count = a["core:sample_count"]

#         power_dbfs = getPower(sample_start, sample_count, rf_lo, rf_hi)
#         if not np.isfinite(power_dbfs):
#             print(f"Invalid frame. This annotation should be dropped.")
#             continue

#         # print(f"Mean integrated power: {power_dbfs:.2f} dBFS")

#         if power_dbfs > minimum_power_dbfs:
#             kept.append(a)

#     kept_data["annotations"] = kept

#     with open(args.outfile, "w") as f:
#         json.dump(kept_data, f, indent=2)
#         print("Wrote to " + args.outfile)






# # -------------------------
# # Output
# # -------------------------
# if __name__ == "__main__":
#     loopAnnotations(args.metafile)
