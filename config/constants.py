HIGH_COUNT_ANTIGENS = ['A0201_ELAGIGILTV_MART-1_Cancer_binder',
					   'A0201_GILGFVFTL_Flu-MP_Influenza_binder',
					   'A0201_GLCTLVAML_BMLF1_EBV_binder',
					   'A0301_KLGGALQAK_IE-1_CMV_binder',
					   'A0301_RLRAEAQVK_EMNA-3A_EBV_binder',
					   'A1101_IVTDFSVIK_EBNA-3B_EBV_binder',
					   'A1101_AVFDRKSDAK_EBNA-3B_EBV_binder',
					   'B0801_RAKFKQLL_BZLF1_EBV_binder']

# Assigns each antigen a unique consistent color
ANTIGEN_COLORS = {'A0201_ELAGIGILTV_MART-1_Cancer_binder': 'tab:blue',
				  'A0201_GILGFVFTL_Flu-MP_Influenza_binder': 'tab:orange',
				  'A0201_GLCTLVAML_BMLF1_EBV_binder': 'tab:brown',
				  'A0301_KLGGALQAK_IE-1_CMV_binder': 'tab:red',
				  'A0301_RLRAEAQVK_EMNA-3A_EBV_binder': 'tab:purple',
				  'A1101_IVTDFSVIK_EBNA-3B_EBV_binder': 'tab:green',
				  'A1101_AVFDRKSDAK_EBNA-3B_EBV_binder': 'tab:pink',
				  'B0801_RAKFKQLL_BZLF1_EBV_binder': 'tab:olive',
				  'no_data': 'tab:grey'}

donor_1_high_count_antigens = ['A1101_IVTDFSVIK_EBNA-3B_EBV_binder',
                               'A0301_KLGGALQAK_IE-1_CMV_binder',
                               'A0201_GILGFVFTL_Flu-MP_Influenza_binder',
                               'A1101_AVFDRKSDAK_EBNA-3B_EBV_binder',
                               'A0201_ELAGIGILTV_MART-1_Cancer_binder']
donor_2_high_count_antigens = ['B0801_RAKFKQLL_BZLF1_EBV_binder',
                               'A0201_GILGFVFTL_Flu-MP_Influenza_binder',
                               'A0301_KLGGALQAK_IE-1_CMV_binder',
                               'A0201_GLCTLVAML_BMLF1_EBV_binder',
                               'A1101_AVFDRKSDAK_EBNA-3B_EBV_binder']

DONOR_SPECIFIC_ANTIGENS = {'1': donor_1_high_count_antigens,
						   '2': donor_2_high_count_antigens,
						   'all': HIGH_COUNT_ANTIGENS}
