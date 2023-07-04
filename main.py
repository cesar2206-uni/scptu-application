# SCPTu Processing (Author: César Sánchez)

# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from matplotlib.ticker import FuncFormatter
from matplotlib import rc
from matplotlib.ticker import MultipleLocator, ScalarFormatter, NullFormatter
import math
import matplotlib as mpl
import matplotlib.font_manager as fm
from matplotlib.ticker import NullFormatter
from datetime import datetime
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import math


# Matplotlib - Configuration
font = fm.FontProperties(family='Century Schoolbook',
                         style='normal')

mpl.rcParams['font.family'] = font.get_name()
mpl.rcParams['font.style'] = font.get_style()

# Some function utils 
def round_up(n, decimals=0):
    """
    Rounding up data to a next decimal - useful to axes in matplotlib
    """
    
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier

def round_down(n, decimals=0):
    """
    Rounding down data to a next decimal - useful to axes in matplotlib
    """
    
    multiplier = 10 ** decimals
    return math.floor(n * multiplier) / multiplier

# Processing
def read_data(path):
    """
    This function with the path of .xlsx, create a list with the data in dataframes
    The order is: [CPTu Data, Velocities Data, Dissipation Data, Pizometric Data, Geology Data and Laboratory Data]
    """
    
    cptu_data = pd.read_excel(path, 
                              sheet_name = "CPTu_data")
    
    dis_data = pd.read_excel(path, 
                             sheet_name = "Dissipation_data",
                             skiprows = 1)
    
    water_table = pd.read_excel(path, 
                                sheet_name = "Dissipation_data",
                                header = None).loc[0, 1]

    piez_data = pd.read_excel(path,
                              sheet_name = "Piezometric_data")
    
    geology_data = pd.read_excel(path,
                                 sheet_name = "Geology_data",
                                 skiprows = 2)
    
    scpt_elevation = pd.read_excel(path,
                              sheet_name = "Geology_data",
                              header = None).loc[0, 1]
    
    borehole_elevation = pd.read_excel(path,
                              sheet_name = "Geology_data",
                              header = None).loc[1, 1]
    
    lab_data = pd.read_excel(path,
                             sheet_name= "Laboratory_data")
    
    vs_data = pd.read_excel(path,
                            sheet_name="Vs_data")
    
    name = path[:-5]
    
    return [cptu_data, dis_data, piez_data, geology_data, lab_data, water_table, scpt_elevation, borehole_elevation, vs_data, name]


def process_dissipation_test(data):
    """
    This function combine the data of cpt and dissipation test + water table
    """
    cpt_data = data[0]
    nf_data = data[5] 
    dis_data = data[1]
    
    # Adding as first value of series the water table
    nf_row = pd.Series([nf_data, 0],
                    index = ["z (m)","PP (m)"]) 

    dis_data = pd.concat([pd.DataFrame(nf_row).T, dis_data],
                        ignore_index=True)
    
    dis_data.drop('WT_assumed', axis=1, inplace=True)
    
    # Extrapolation of water table
    k = 0
    
    if cpt_data["z (m)"].iloc[-1] > dis_data["z (m)"].iloc[-1]:
        
        ## Extrapolation ~ Hidrostatic Case of WT only
        if len(dis_data) == 1:
            hi_row = pd.Series([cpt_data["z (m)"].iloc[-1], 
                                cpt_data["z (m)"].iloc[-1] - nf_data], 
                            index=["z (m)", "PP (m)"])      
            dis_data = pd.concat([dis_data, pd.DataFrame(hi_row).T], 
                                ignore_index=True)
            k = 1
        
        ## Extrapolation ~ Last dissipation test below the last value of CPT data  
        else:
            ex_row = pd.Series([cpt_data["z (m)"].iloc[-1], 
                                dis_data["PP (m)"].iloc[-1] 
                                + cpt_data["z (m)"].iloc[-1] 
                                - dis_data["z (m)"].iloc[-1]], 
                            index=["z (m)", "PP (m)"])          
            dis_data = pd.concat([dis_data, 
                                pd.DataFrame(ex_row).T], 
                                ignore_index=True)
            
            k = 1
    
    # Dissipation test above cpt data is necessary to delete if we want a continuous data     
    row_del = len(dis_data[dis_data["z (m)"] > cpt_data["z (m)"].iloc[-1]])                 
    
    # Merge the CPTu data with dissipation data
    cpt_data = pd.merge(cpt_data, 
                    dis_data, 
                    on = "z (m)", 
                    how = "outer")
    cpt_data = cpt_data.sort_values(by = "z (m)")
    cpt_data.reset_index(drop=True,inplace=True)
    
    # Interpolation of dissipation data
    cpt_data.set_index('z (m)',
                    inplace = True)
    cpt_data.interpolate(method = 'index',
                        inplace = True,
                        limit_direction = 'forward')
    cpt_data.reset_index(inplace = True)
    
    # PP before the WT equal to zero
    cpt_data.loc[cpt_data["z (m)"] < dis_data["z (m)"].iloc[0], "PP (m)"] = 0
    
    # Transforming the PP (m) to kPa
    cpt_data["u0 (kPa)"] = cpt_data["PP (m)"] * 9.81
    
    # Saving the new data
    data[0] = cpt_data
    
    return data


def pre_graph_dissipation_test(data, y_limit, ax):
    """
    This function prepare the axes of graph of dissipation test based in data
    data: Data + dissipation process
    y_limit: Limiting the y axis
    """
    # Data used
    cpt_data = data[0]
    dis_data = data[1]
    piez_data = data[2]
    geology_data = data[3]
    nf_data = data[5] 
    
    # Plot the data Interpolated of dissipation tests
    ax.plot(cpt_data["u0 (kPa)"], 
            cpt_data["z (m)"],
            label = "Data interpolada",
            color ="#7EBCCB")

    # Add all disipation tests
    ax.plot(dis_data["PP (m)"] * 9.81,
        dis_data["z (m)"], 
        "o", 
        label = "Ensayos de disipación \n eq", 
        color = "#519EB4")
    
    # Add all disipation tests WT_assumed
    ax.plot(dis_data["PP (m)"][dis_data["WT_assumed"] == True] * 9.81,
            dis_data["z (m)"][dis_data["WT_assumed"] == True], 
            "o", 
            label = "Ensayos de disipación \n asumidos", 
            color = "salmon")
    
    # Add Water Table
    ax.plot(0,
        nf_data,
        "o", 
        color = "#163824",
        label = "NF = " 
        + str(round(nf_data, 2)) 
        + " m")
    
    # Add Piezometric Data
    ax.plot(piez_data["PP (m)"] * 9.81,
        piez_data["z (m)"],
        "o",
        color = "#28712B")
    
    # Adding a non-important point in order to appear a label in legend
    ax.plot(-20,
        -20,
        "o",
        color = "#28712B",
        label = "Piezómetros")
    
    # Adding text in piezometric data
    for i, row in piez_data.iterrows():
        if row["z (m)"] < float(y_limit):
                ax.text(row["PP (m)"] * 9.81 + 2,
                        row["z (m)"] - 1, 
                        row["Piezometer"], 
                        ha='center', 
                        va = "bottom", 
                        color = "#28712B", 
                        fontsize = 10)
         
    # Obtaining the x limit, in case no piezometric data, use the max of cpt       
    try:        
        x_limit = max(max(cpt_data["u0 (kPa)"]), max(piez_data["PP (m)"] * 9.81)) * 1.2
    except:
        x_limit = max(cpt_data["u0 (kPa)"]) * 1.2
    
    # Plotting divisions of geology  
    for i, row in geology_data.iterrows():
        ax.plot(
            [0, x_limit],
            [row["z (m)"], row["z (m)"]],
            "--",
            color = "#BA8800"
        )
        ax.text(x_limit /2,
                row["z (m)"],
                row["Text"],
                ha='center', 
                va = "top", 
                color = "#BA8800", 
                fontsize = 10)
    
    # Legend and grid
    ax.legend(loc = "upper right")
    ax.grid(True, 
        ls="-",
        color='0.9')
        
    # Ticks of graph
    ax.tick_params(axis="x",
                direction="in",
                length=6)
    ax.tick_params(axis="y",
                direction="in",
                length=6)
    ax.xaxis.set_minor_locator(MultipleLocator(20))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(axis="y", which='minor',direction="in", length=3)
    ax.tick_params(axis="x", which='minor',direction="in", length=3)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')

    # Limits of graph
    ax.set_xlim([0, x_limit])
    ax.set_ylim(0, float(y_limit))
    
    # Labels    
    ax.set_xlabel('Presión de Poros (kPa)') 
    ax.set_ylabel('Profundidad (m)') 
    ax.set_title("Interpolación Ensayo de Disipación \n" + data[-1])
    ax.invert_yaxis()
    
  
    return 0


def graph_dissipation_test(data, y_limit, option = 0):
    """
    This function create a plot of dissipation test based in data
    data: Data + dissipation process
    y_limit: Limiting the y axis
    """
    fig, ax = plt.subplots(nrows = 1,
                       ncols = 1,
                       figsize = (5, 12))

    pre_graph_dissipation_test(data, y_limit, ax)
    
    # Option 1: Save the plot as svg
    if option == 1:
        time_now = datetime.now()
        current_time = time_now.strftime("%Y-%m-%d %H_%M")
        fig.savefig("Plot Dissipation-" + data[-1] + "_" + current_time + ".svg")
        return 0

    if option == 2:
        time_now = datetime.now()
        current_time = time_now.strftime("%Y-%m-%d %H_%M")
        data[0].to_excel("results" + data[-1] +"_" + current_time + ".xlsx", index = False)     
        return 0
        
    plt.show()   

    return 0

def pre_graph_ip(data, y_limit, ax):
    """
    This function create the ip graph based in lab
    data: Normal Data (no necessary dissipation test process)
    """
    # Data used
    data_ip = data[4]
    data_i_ip = data_ip[pd.to_numeric(data_ip["IP"], errors='coerce').notnull()]
    cota_scpt = data[6]
    cota_bh = data[7]
    borehole_name = str(data_ip["Borehole"].iloc[-1])

    # Scatter plot of IP vs Elevation
    ax.plot(
        data_i_ip["IP"],
        float(cota_scpt) - (float(cota_bh) - data_i_ip["Mean"]),
        "o",
        color="darkblue",
        label="IP"
    )

    # Line in IP = 12
    ax.plot(
        [12, 12], 
        [0, float(y_limit)],
        "--", 
        color = "gray",
        label = "IP = 12"
    )

    # Text of IP = 12
    ax.text(12*1.2, float(y_limit)-1, 'IP = 12', color = "gray")


    # Limits of graphs
    ax.set_xlim(0,
                max(20, round_up(max(data_i_ip["IP"])), -1)
                )
    ax.set_ylim(0, float(y_limit))
    
    # Labels of graph
    ax.set_ylabel("Profundidad (m)")
    ax.set_xlabel('Indice Plástico (%)') 
    ax.set_title("Indice Plástico (IP)\n" + borehole_name)

    # Ticks
    ax.tick_params(axis="x",
                direction="in",
                length=6)
    ax.tick_params(axis="y",
                direction="in",
                length=6)
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(axis="y", which='minor',direction="in", length=3)
    ax.tick_params(axis="x", which='minor',direction="in", length=3)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')

    # Grid and Inversion of yaxis
    ax.grid(True, 
            ls="-",
            color='0.9')
    ax.invert_yaxis()
    
    return 0


def graph_ip(data, y_limit, option = 0):
    """
    This function create a plot of IP vs elevation based in data
    data: Data + dissipation process
    y_limit: Limiting the y axis
    """
    fig, ax = plt.subplots(nrows = 1,
                    ncols = 1,
                    figsize = (5, 12)) 
    
    pre_graph_ip(data, y_limit, ax)
    
    # Option 1: Save the plot as svg
    if option == 1:
        time_now = datetime.now()
        current_time = time_now.strftime("%Y-%m-%d %H_%M")
        fig.savefig("Plot IP-" + data[-1] + "_" + current_time + ".svg")
        return 0
    
    plt.show()   

    return 0  
    

def graph_ip_dissipation(data, y_limit, option = 0):
    """
    This function create the IP and dissipation graph based in lab
    data: Dissipation Processed Data (no necessary dissipation test process)
    """ 
    fig, axes = plt.subplots(nrows = 1,
                       ncols = 2,
                       figsize = (10, 12), sharey=True)
    
    
    pre_graph_ip(data, y_limit, axes[0])
    pre_graph_dissipation_test(data, y_limit, axes[1])

    axes[1].set_ylabel('') 
    # Option 1: Save the plot as svg
    if option == 1:
        time_now = datetime.now()
        current_time = time_now.strftime("%Y-%m-%d %H_%M")
        fig.savefig("Plot IP and Dissipation_" + data[-1] + "_" + current_time + ".svg")
        return 0

    if option == 2:
        time_now = datetime.now()
        current_time = time_now.strftime("%Y-%m-%d %H_%M")
        data[0].to_excel("results" + data[-1] +"_" + current_time + ".xlsx", index = False)     
        return 0
    
    plt.show()
        
    return 0

def process_sbt(data):
    """
    This function process the data and provide data to make SBT plots
    """
     # Data used
    cpt_data = data[0]
    
    # Tip corrected resistance
    a = 0.8 
    cpt_data["qt (MPa)"] = cpt_data["qc (Mpa)"] + (1 - a) * cpt_data["u2 (kPa)"] / 1000
    
    # Friction Ratio
    cpt_data["Rf (%)"] = cpt_data["fs (kPa)"] * 100 / (cpt_data["qt (MPa)"] * 1000)
    
    # Unit Weight Estimated
    gamma_w = 9.81  # KN/m3
    gamma_min = 13.72  # KN/m3
    Gs = 2.65
    Pa = 101.325  # kPa

    def gamma_estimated(row):
        if row["qt (MPa)"] > 0:
            if row["fs (kPa)"] >0:
                val = (0.27 * np.log10(row["Rf (%)"]) 
                    + 0.36 * np.log10(row["qt (MPa)"] * 1000 / Pa) # Fr (%) Calculation
                    + 1.236) * (Gs / 2.65) * gamma_w
            else: 
                val = gamma_min
        else:
            val = gamma_min
        return val

    cpt_data["gamma_est (kN/m3)"] = cpt_data.apply(gamma_estimated,
                                                axis=1)
    
    # Vertical Effective Stress and Vertical Stress
    cpt_data["sigma_v (kPa)"] = (cpt_data["z (m)"].diff().fillna(cpt_data["z (m)"]) * cpt_data["gamma_est (kN/m3)"]).cumsum()
    cpt_data["sigma_v_0 (kPa)"] = cpt_data["sigma_v (kPa)"] - cpt_data["u0 (kPa)"]
    
    # Fr (Friction ratio considering the vertical stress)
    cpt_data["Fr (%)"] = cpt_data["fs (kPa)"] * 100 / (cpt_data["qt (MPa)"] * 1000 - cpt_data["sigma_v (kPa)"])
    
    # SBTn Robertson 2016   
    def sbt_func(zGuess, *Params):
        Ic, n = zGuess
        sigma_v, sigma_v_o, qt, Fr = Params
        eq_1 = min(0.381 * Ic + 0.05 * sigma_v_o / Pa - 0.15, 1) - n
        Cn = (Pa / sigma_v_o) ** n
        Qt_n = (qt * 1000- sigma_v) * Cn / Pa
        if Qt_n < 0: 
            Qt_n = 0
        if Fr == 0:
            eq_3 = Ic - 1
        else:
            eq_3 = ((3.47 - np.log10(Qt_n)) ** 2 + (np.log10(Fr) + 1.22) ** 2) ** 0.5 - Ic
        return eq_1, eq_3

    # Solve the Ic and n bidirectional problem
    cpt_data['I_c'], cpt_data['n'] = zip(*cpt_data.apply(
        lambda x: opt.fsolve(
            sbt_func,
            np.array([0.25, 0.25]),
            args=(
                x["sigma_v (kPa)"],
                x["sigma_v_0 (kPa)"],
                x["qt (MPa)"],
                x["Fr (%)"]
                )
            )
        , 1)
        )   

    # Recalculate the other variables
    cpt_data['Cn'] = (Pa / cpt_data["sigma_v_0 (kPa)"]) ** cpt_data["n"]
    cpt_data["Q_tn"] = (cpt_data["qt (MPa)"] * 1000 - cpt_data["sigma_v (kPa)"]) * cpt_data["Cn"] / Pa
    cpt_data["Q_tn"][cpt_data["Q_tn"] < 0] = 0
    
    # Clasificación : Contactive - Dilative (CD) 
    cpt_data["CD"] = (cpt_data["Q_tn"] - 11) * ((1 + 0.06 * cpt_data["Fr (%)"]) ** 17)
    
    # SBT type index IB
    cpt_data["I_B"] = 100 * (cpt_data["Q_tn"] + 10) / (cpt_data["Q_tn"] * cpt_data["Fr (%)"] + 70)
    cpt_data["I_B"][cpt_data["Q_tn"] == 0] = 0
    
    # Obtain the Bq
    cpt_data["B_q"] = (cpt_data["u2 (kPa)"] - cpt_data["u0 (kPa)"])/(cpt_data["qt (MPa)"] * 1000 - cpt_data["sigma_v (kPa)"])
    cpt_data["B_q"][cpt_data["u2 (kPa)"] <= 0] = 0
    # cpt_data["B_q"] = 0

    
    
    data[0] = cpt_data
    
    
    
    
    return data


def lab_cpt_process(data):
    """
    A function which provide the combination of lab and cpt_data in xlsx
    """    
    cpt_data = data[0]
    cota_scpt = data[6]
    cota_bh = data[7]
    lab_data = data[4]

    lab_data["z (m)"] = float(cota_scpt) - (float(cota_bh) - lab_data["Mean"])

    lab_cpt_data = pd.merge_asof(lab_data.sort_values('z (m)'), cpt_data.sort_values('z (m)'), direction='nearest',on='z (m)')
    lab_cpt_data.loc[lab_cpt_data["I_c"] > 2.6, "SB_Ic"] = "Clay-Like"
    lab_cpt_data.loc[lab_cpt_data["I_c"] <= 2.6, "SB_Ic"] = "Sand-Like"
    
    return lab_cpt_data

def Ic_lab_data(data):
    """
    This function create a graph of comparison between SUCS clasification and behaviour clasification (Ic)
    """
        
    cpt_data = data
    cpt_data = cpt_data.drop(index=cpt_data[cpt_data['SUCS'] == "-"].index)
    
    # Creating the plot
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (9, 9))

    
    # Divisions of plot - Ic
    ax.plot(
        np.linspace(0.1, 10, 200),
        10 ** (3.47 - (2.6 ** 2 - (np.log10(np.linspace(0.1, 10, 200)) + 1.22) ** 2)**0.5),
        "--",
        color = "black"
    )    
    
    ax.plot(
        np.linspace(0.1, 10, 200),
        10 ** (3.47 - (2.9 ** 2 - (np.log10(np.linspace(0.1, 10, 200)) + 1.22) ** 2)**0.5),
        "--",
        color = "black"
    )
    
    ax.plot(
        np.linspace(0.1, 10, 200),
        10 ** (3.47 - (2.7 ** 2 - (np.log10(np.linspace(0.1, 10, 200)) + 1.22) ** 2)**0.5),
        "--",
        color = "black"
    )
    
    ax.plot(
        np.linspace(0.1, 10, 200),
        10 ** (3.47 - (2.8 ** 2 - (np.log10(np.linspace(0.1, 10, 200)) + 1.22) ** 2)**0.5),
        "--",
        color = "black"
    )
    
    # Color of the Behaviour zones
    ax.fill_between(np.linspace(0.1, 10, 200),
                    10 ** (3.47 - (2.6 ** 2 - (np.log10(np.linspace(0.1, 10, 200)) + 1.22) ** 2)**0.5),
                    1000,
                    color="green",
                    alpha=0.2
                    )
    
    ax.fill_between(np.linspace(0.1, 10, 200),
                    10 ** (3.47 - (2.9 ** 2 - (np.log10(np.linspace(0.1, 10, 200)) + 1.22) ** 2)**0.5),
                    10 ** (3.47 - (2.6 ** 2 - (np.log10(np.linspace(0.1, 10, 200)) + 1.22) ** 2)**0.5),
                    color="blue",
                    alpha=0.2
                    )
    
    ax.fill_between(np.linspace(0.1, 10, 200),
                    1,
                    10 ** (3.47 - (2.9 ** 2 - (np.log10(np.linspace(0.1, 10, 200)) + 1.22) ** 2)**0.5),
                    color="red",
                    alpha=0.2
                    )       
    
    # Text in the graph
    ax.text(0.5, 3, 'Clay-Like', ha='center', va = "center", fontsize = 22,color = "black", fontweight='bold')
    ax.text(0.5, 2, '(Ic ≥ 2.9)', ha='center', va = "center", fontsize = 18,color = "black", fontstyle='italic')
    ax.text(2.2, 500, 'Sand-Like', ha='center', va = "center", fontsize = 22,color = "black", fontweight='bold')
    ax.text(2.2, 350, '(Ic ≤ 2.6)', ha='center', va = "center", fontsize = 18,color = "black", fontstyle='italic') 
    ax.text(0.3, 50, 'Transition', ha='center', va = "center", fontsize = 22,color = "black", fontweight='bold')
    ax.text(0.2, 24, 'Ic = 2.6', ha='center', va = "center", fontsize = 18,color = "black", fontstyle='italic') 
    ax.text(0.16, 12, 'Ic = 2.9', ha='center', va = "center", fontsize = 18,color = "black", fontstyle='italic')    
    
    # Creation the arrows
    plt.annotate('', xy=(1, 12), xytext=(0.6, 40), 
                arrowprops=dict(facecolor='black', width=0.1, headwidth = 5, headlength = 5),
                )
    ax.plot([0.15, 0.6], [40, 40], color = "black")

    plt.annotate('', xy=(0.4, 10), xytext=(0.3, 20), 
                arrowprops=dict(facecolor='black', width=0.1, headwidth = 5, headlength = 5),
                )
    ax.plot([0.15, 0.3], [20, 20], color = "black")

    plt.annotate('', xy=(0.37, 4.6), xytext=(0.23, 10.5), 
                arrowprops=dict(facecolor='black', width=0.1, headwidth = 5, headlength = 5),
                )
    ax.plot([0.11, 0.23], [10.5, 10.5], color = "black")
    
    # Dictionaries to asign a color and a sign to type of sols
    markers_style = {'MH': 'o', 'ML': 's', "CH": "p", "CL": "P", "CL-ML":"*",
               'SC': '^', "SW-SC": "h", "SP-SM": "D", "SC-SM":"<", "SM": "v",
               "GW-GC": ">", "GP-GC": "d", "GW": "H", "GP-GM":"X", "GC":10, "GM":11}
    
    markers_color = {'MH': '#8B0000', 'ML': '#FF5733', "CH": "#FFC0CB", "CL": "#DC143C", "CL-ML":"#8B0000",
               'SC': '#6B8E23', "SW-SC": "#008080", "SP-SM": "#32CD32", "SC-SM":"#ADFF2F", "SM": "#9ACD32",
               "GW-GC": "#556B2F", "GP-GC": "#00FA9A", "GW": "#008000", "GP-GM":"#228B22", "GC":"#2E8B57", "GM":"#2E8B"}
    
    # Plotting the points
    for label, group in cpt_data.groupby('SUCS'):
        ax.scatter(group["Fr (%)"], group["Q_tn"], s = 50, marker=markers_style[label], edgecolor='black', color = markers_color[label],  label=label)
    
    ## Other settings
    ax.set_yscale('log')
    ax.set_xscale('log')

    for axis in [ax.xaxis, ax.yaxis]:
        formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))
        axis.set_major_formatter(formatter)

    ax.set_xlim(0.1, 10)
    ax.set_ylim(1, 1000)
    ax.grid(True, which="both", ls="-", color='0.9')
    ax.set_axisbelow(True)

    ax.set_xlabel("Normalized friction ratio, $F_r (\% )$", fontsize = 16)
    ax.set_ylabel("Normalized cone resistance, $Q_{tn}$", fontsize = 16)
    ax.legend(loc = "upper left", ncol=2)
    plt.show()
    
    return 0

def Properties_Profiles(data):
    """"
    A Function to plot all properties profiles
    """
    
    data_i = data[4]
    data_scpt_i = data[0]
    cota_scpt = data[6]
    cota_bh = data[7]    
    scpt_name = data[-1]
    bh_name = str(data_i["Borehole"].iloc[-1])
    data_div_i = data[3]

    color ="blue"
    # Reading and preprocessing data
    ## Reading data of Boreholes
    # data = pd.read_excel("data.xlsx")
    # data_div = pd.read_excel("div.xlsx")
    # data_cotas = pd.read_excel("cotas.xlsx")
    # data_scpt_i = pd.read_excel("./SCPT/cpt_data_ausenco.xlsx", sheet_name= scpt_name)


    ## Preprocesssing result data of tests
    
    data_i["Cota_prom"] = float(cota_bh) - data_i["Mean"]
    
    data_div_i["Cota"] = float(cota_scpt) - data_div_i["z (m)"]
    
    
    data_i_ip = data_i.drop(index=data_i[data_i['IP'] == "-"].index)
    data_i_w = data_i.drop(index=data_i[data_i['w(%)'] == "-"].index)
    data_i_wll = data_i_w.drop(index=data_i_w[data_i_w['LL'] == "-"].index)
    data_i_wll = data_i_wll[pd.to_numeric(data_i_wll["LL"], errors='coerce').notnull()]
    
    data_i_fi = data_i.drop(index=data_i[data_i['Total'] == "-"].index)
    data_i_ga = data_i.drop(index=data_i[data_i['% Gravel'] == "-"].index)
    data_i_sa = data_i.drop(index=data_i[data_i['% Sand'] == "-"].index)
    data_i_sucs = data_i.drop(index=data_i[data_i['SUCS'] == "-"].index)
    ## Preprocessing geology data 
    # data_div_i = data_div.loc[data_div.Sondaje == bh_name].loc[data_div.SCPT == scpt_name]

    ## Preprocessing scpt data
    cota_i = cota_scpt
    data_scpt_i["Cota (m)"] = float(cota_i) - data_scpt_i["z (m)"]

    # Plot
    fig, ax = plt.subplots(nrows = 1, ncols = 8, sharey = True, figsize = (24, 12), gridspec_kw={'width_ratios': [3, 3, 3, 3, 3, 3, 3, 1]})

    ## Elevation (m.s.n.m.)

    ax[0].yaxis.set_minor_locator(MultipleLocator(1))
    ax[0].set_ylim(
        round_down(min(data_scpt_i["Cota (m)"]), -1),
        round_up(max(data_scpt_i["Cota (m)"]), -1)
    )

    ## SCPT Plot
    ### plot
    ax[0].plot(
        data_scpt_i["qc (Mpa)"],
        data_scpt_i["Cota (m)"],
        "-",
        color = color,
        label = "qc (Mpa)"
    )

    ### Limits
    limits = []
    limits.append([0,  max(data_scpt_i["qc (Mpa)"]) * 1.2])
    ax[0].set_xlim(limits[0])

    ### Ticks
    ax[0].tick_params(axis="x",direction="in", length=6)
    ax[0].tick_params(axis="y",direction="in", length=6)
    ax[0].xaxis.set_minor_locator(MultipleLocator(5))
    ax[0].tick_params(axis="y", which='minor',direction="in", length=3)
    ax[0].tick_params(axis="x", which='minor',direction="in", length=3)
    ax[0].yaxis.set_ticks_position('both')
    ax[0].xaxis.set_ticks_position('both')

    ### Titles and grid
    # ax[0].title.set_text('qc (MPa) vs Elevación \n' + "SCPTu-" + scpt_name[-3:])
    ax[0].title.set_text('qc (MPa) vs Elevación \n' + scpt_name)

    ax[0].title.set_size(11)
    ax[0].grid()

    ## % Fines content vs Elevation

    ### Plot
    ax[1].plot(data_i_fi["Total"], 
            data_i_fi["Cota_prom"], 
            "o", 
            color = color,
            label = "% Finos"
            )

    ### Limits and contacts

    limits.append([0, 100])
    ax[1].set_xlim(limits[1])

    ### Ticks
    ax[1].tick_params(axis="x",direction="in", length=6)
    ax[1].tick_params(axis="y",direction="in", length=6)
    ax[1].xaxis.set_minor_locator(MultipleLocator(5))
    ax[1].tick_params(axis="y", which='minor',direction="in", length=3)
    ax[1].tick_params(axis="x", which='minor',direction="in", length=3)
    ax[1].yaxis.set_ticks_position('both')
    ax[1].xaxis.set_ticks_position('both')

    ### Titles and grid
    ax[1].title.set_text('% Finos vs Elevación \n' + bh_name)
    ax[1].title.set_size(11)
    ax[1].grid()

    ## % W vs Elevation

    ### Plot
    ax[2].plot(data_i_w["w(%)"], 
            data_i_w["Cota_prom"], 
            "o", 
            color = color,
            label = "w (%)"
            )

    ### Limits

    limits.append([0, round_up((max(data_i_w["w(%)"])), -1)])
    ax[2].set_xlim(limits[2])

    ### Ticks
    ax[2].tick_params(axis="x",direction="in", length=6)
    ax[2].tick_params(axis="y",direction="in", length=6)
    ax[2].xaxis.set_minor_locator(MultipleLocator(5))
    ax[2].tick_params(axis="y", which='minor',direction="in", length=3)
    ax[2].tick_params(axis="x", which='minor',direction="in", length=3)
    ax[2].yaxis.set_ticks_position('both')
    ax[2].xaxis.set_ticks_position('both')

    ### Titles and grid
    ax[2].title.set_text('% W vs Elevación \n' + bh_name)
    ax[2].title.set_size(11)
    ax[2].grid()


    ## % Grava vs Elevation

    ### Plot
    ax[3].plot(data_i_ga["% Gravel"], 
            data_i_ga["Cota_prom"], 
            "o", 
            color = color,
            label = "Grava (%)"
            )

    ### Limits
    limits.append([0, 100])
    ax[3].set_xlim(limits[3])

    ### Ticks
    ax[3].tick_params(axis="x",direction="in", length=6)
    ax[3].tick_params(axis="y",direction="in", length=6)
    ax[3].xaxis.set_minor_locator(MultipleLocator(5))
    ax[3].tick_params(axis="y", which='minor',direction="in", length=3)
    ax[3].tick_params(axis="x", which='minor',direction="in", length=3)
    ax[3].yaxis.set_ticks_position('both')
    ax[3].xaxis.set_ticks_position('both')

    ### Titles and grid
    ax[3].title.set_text('% Grava vs Elevación \n' + bh_name)
    ax[3].title.set_size(11)
    ax[3].grid()

    ## IP vs Elevation

    ### Plot
    ax[4].plot(
        data_i_ip["IP"], 
        data_i_ip["Cota_prom"], 
        "o", 
        color = color,
        label = "IP"
    )

    ax[4].plot(
        [12, 12], 
        [round_down(min(data_scpt_i["Cota (m)"]), -1), round_up(max(data_scpt_i["Cota (m)"]), -1)],
        "--", 
        color = "gray",
        label = "IP = 12"
    )

    ax[4].text(12*1.2, round_down(min(data_scpt_i["Cota (m)"]), -1) + 5, 'IP = 12', color = "gray")

    ### Limits
    try: 
        limits.append([0, round_up((max(data_i_ip["IP"])), -1)])
    except:
        limits.append([0, 100])

    ax[4].set_xlim(limits[4])

    ### Ticks
    ax[4].tick_params(axis="x",direction="in", length=6)
    ax[4].tick_params(axis="y",direction="in", length=6)
    ax[4].xaxis.set_minor_locator(MultipleLocator(5))
    ax[4].tick_params(axis="y", which='minor',direction="in", length=3)
    ax[4].tick_params(axis="x", which='minor',direction="in", length=3)
    ax[4].yaxis.set_ticks_position('both')
    ax[4].xaxis.set_ticks_position('both')

    ### Titles and grid
    ax[4].title.set_text('Indice Plástico vs Elevación \n' + bh_name)
    ax[4].title.set_size(11)
    ax[4].grid()

    ## % Arena vs Elevation

    ### Plot
    ax[5].plot(data_i_sa["% Sand"], 
            data_i_sa["Cota_prom"], 
            "o", 
            color = color,
            label = "Arena (%)"
            )

    ### Limits
    limits.append([0, 100])
    ax[5].set_xlim(limits[5])

    ### Ticks
    ax[5].tick_params(axis="x",direction="in", length=6)
    ax[5].tick_params(axis="y",direction="in", length=6)
    ax[5].xaxis.set_minor_locator(MultipleLocator(5))
    ax[5].tick_params(axis="y", which='minor',direction="in", length=3)
    ax[5].tick_params(axis="x", which='minor',direction="in", length=3)
    ax[5].yaxis.set_ticks_position('both')
    ax[5].xaxis.set_ticks_position('both')

    ### Titles and grid
    ax[5].title.set_text('% Arena vs Elevación \n' + bh_name)
    ax[5].title.set_size(11)
    ax[5].grid()

    ## % Indice de Liquidez vs Elevation

    ### Plot
    ax[6].plot(
        data_i_wll["w(%)"] / data_i_wll["LL"], 
        data_i_wll["Cota_prom"], 
        "o", 
        color = color,
        label = "w/LL"
    )

    ### Limits
    try:    
        limits.append([0, max(data_i_wll["w(%)"] / data_i_wll["LL"]) * 1.5])
    except:
        limits.append([0, 4])
    ax[6].set_xlim(limits[6])

    ### Ticks
    ax[6].tick_params(axis="x",direction="in", length=6)
    ax[6].tick_params(axis="y",direction="in", length=6)
    ax[6].xaxis.set_minor_locator(MultipleLocator(5))
    ax[6].tick_params(axis="y", which='minor',direction="in", length=3)
    ax[6].tick_params(axis="x", which='minor',direction="in", length=3)
    ax[6].yaxis.set_ticks_position('both')
    ax[6].xaxis.set_ticks_position('both')

    ### Titles and grid
    ax[6].title.set_text('Indice de Liquidez vs Elevación \n' + bh_name)
    ax[6].title.set_size(11)
    ax[6].grid()


    ## SUCS plot

    for i, row in data_i_sucs.iterrows():
        if row["Cota_prom"] > round_up(max(data_scpt_i["Cota (m)"]), -1):
            pass
        elif row["Cota_prom"] < round_down(min(data_scpt_i["Cota (m)"]), -1):
            pass
        else:
            ax[7].text(0.5, row["Cota_prom"], row["SUCS"], ha='center', va = "center", color = color)
            ax[7].add_patch(patches.Rectangle((0, float(cota_bh) - row["To"]), 1, row["To"] - row["From"], facecolor = "white", edgecolor = "silver", hatch = "///"))

    ax[7].title.set_text('Perfil de Suelo \n' + bh_name)
    ax[7].title.set_size(11)
    ax[7].get_xaxis().set_visible(False)
    ax[7].get_yaxis().set_visible(False)


    ## Plotting texts and lines

    for i in range(1, len(ax)-1):
        for j, row in data_div_i.iterrows():
            ax[i].plot(limits[i],[row["Cota"], row["Cota"]], "--", color = "darkgoldenrod")
            ax[i].text((limits[i][0] + limits[i][1])/2, row["Cota"] , row["Text"] + "\n" + str(round(row["Cota"], 2)) + " m.s.n.m", ha='center', va = "center", color = "darkgoldenrod")


    
    time_now = datetime.now()
    current_time = time_now.strftime("%Y-%m-%d %H_%M")
    fig.savefig("Plot Properties Profile_" + data[-1] + "_" + current_time + ".svg")

    return 0   
    

def SBTn_Robertson2016(data):
    """
    
    """
    cpt_data = data
    
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (8, 9))

    # Parititions of SBT plot   
    ax.plot([0.1, 2], [10, 10], "-", color = "black")
    ax.plot([2, 2], [1, 10], "-", color = "black")

    ax.plot(
        np.linspace(0.1, 10, 200),
        (70 / ((1 + 0.06 * np.linspace(0.1, 10, 200)) ** 17)) + 11,
        "-",
        color = "black"
    )

    ax.plot(
        np.linspace(2, 4.5454545454, 200),
        (22 * 70 - 1000)/(100 - np.linspace(2, 4.5454545454, 200) * 22),
        "-",
        color = "black"
    )

    ax.plot(
        np.linspace(0.1, 3.125, 200),
        (32 * 70 - 1000)/(100 - np.linspace(0.1, 3.125, 200) * 32),
        "-",
        color = "black"
    )

    ax.text(0.5, 3.2, 'CCS', ha='center', va = "center", fontsize = 22,color = "black")
    ax.text(4.5, 3.2, 'CC', ha='center', va = "center", fontsize = 22,color = "black")
    ax.text(6, 25, 'CD', ha='center', va = "center", fontsize = 22,color = "black")
    ax.text(1.5, 15, 'TC', ha='center', va = "center", fontsize = 22,color = "black")
    ax.text(0.35, 25, 'SC', ha='center', va = "center", fontsize = 22,color = "black")
    ax.text(0.6, 150, 'SD', ha='center', va = "center", fontsize = 22,color = "black")
    ax.text(2.75, 30, 'TD', ha='center', va = "center", fontsize = 22,color = "black")
    ax.text(2.2, 500, "$I_B=32$", ha='center', va = "center", fontsize = 16,color = "black")
    ax.text(6, 200, "$I_B=22$", ha='center', va = "center", fontsize = 16,color = "black")
    ax.text(0.14, 90, "$CD = 70$", ha='center', va = "center", fontsize = 16,color = "black")

    ax.scatter(cpt_data["Fr (%)"], cpt_data["Q_tn"], s = 5,  c = cpt_data["z (m)"], cmap='jet', alpha=0.5)
    ax.set_yscale('log')
    ax.set_xscale('log')

    for axis in [ax.xaxis, ax.yaxis]:
        formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))
        axis.set_major_formatter(formatter)

    ax.set_xlim(0.1, 10)
    ax.set_ylim(1, 1000)
    im = ax.scatter(cpt_data["Fr (%)"], cpt_data["Q_tn"], s = 5,  c = cpt_data["z (m)"], cmap='jet', alpha=0.5)
    ax.grid(True, which="both", ls="-", color='0.9')
    cbar = plt.colorbar(im, orientation='horizontal', fraction=0.1, pad=0.1)
    cbar.set_label("Profundidad (m)")

    ax.set_xlabel("$F_r (\% )$", fontsize = 18)
    ax.set_ylabel("$Q_{tn}$", fontsize = 20)
    ax.set_title("SBTn - Robertson (2016) \n" , fontsize = 16)
    # plt.tight_layout()
    plt.show()
    
    return 0

def PDF_Disipation(data):
    """"
    A function to obtain minimalistic dissipation test ~ ready to convert in dxf - Via AutoCAD
    """
    
    # Data used
    cpt_data = data[0]
    dis_data = data[1]
    piez_data = data[2]
    geology_data = data[3]
    nf_data = data[5] 
    
    def set_size(w,h, ax=None):
        """ w, h: width, height in inches """
        if not ax: ax=plt.gca()
        l = ax.figure.subplotpars.left
        r = ax.figure.subplotpars.right
        t = ax.figure.subplotpars.top
        b = ax.figure.subplotpars.bottom
        figw = float(w)/(r-l)
        figh = float(h)/(t-b)
        ax.figure.set_size_inches(figw, figh)
        
    # Plot the data Interpolated of dissipation tests
    
    # Obtaining the x limit, in case no piezometric data, use the max of cpt       
    try:        
        x_limit = max(max(cpt_data["u0 (kPa)"]), max(piez_data["PP (m)"] * 9.81)) * 1.2
    except:
        x_limit = max(cpt_data["u0 (kPa)"]) * 1.2

    y_limit = round_up(max(cpt_data["z (m)"])* 1.2, -1)
    
    fig, ax = plt.subplots(nrows = 1,
                       ncols = 1)
    #                    figsize = (2, y_limit/6))
    set_size(2, y_limit/6)
    
    ax.plot(cpt_data["u0 (kPa)"], 
            cpt_data["z (m)"],
            label = "Data interpolada",
            color ="#7EBCCB")

    # Add all disipation tests
    ax.plot(dis_data["PP (m)"] * 9.81,
        dis_data["z (m)"], 
        "o", 
        label = "Ensayos de disipación \n eq", 
        color = "#519EB4")
    
    # Add all disipation tests WT_assumed
    ax.plot(dis_data["PP (m)"][dis_data["WT_assumed"] == True] * 9.81,
            dis_data["z (m)"][dis_data["WT_assumed"] == True], 
            "o", 
            label = "Ensayos de disipación \n asumidos", 
            color = "salmon")
    
    # Add Water Table
    ax.plot(0,
        nf_data,
        "o", 
        color = "#163824",
        label = "NF = " 
        + str(round(nf_data, 2)) 
        + " m")
    
    
    # Plotting divisions of geology  
    for i, row in geology_data.iterrows():
        ax.plot(
            [0, x_limit],
            [row["z (m)"], row["z (m)"]],
            "--",
            color = "#BA8800"
        )
    
    # Legend and grid
    # ax.legend(loc = "upper right")
    ax.grid(True, 
        ls="-", which="both",
        color='black')
    # plt.minorticks_on()
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    
    ax.tick_params(axis="y", which='minor',direction="in", length=3)
    # ax.xaxis.set_minor_locator(None)
    # Ticks of graph
    ax.tick_params(axis="x",
                direction="in",
                length=6)
    ax.tick_params(axis="y",
                direction="in",
                length=6)
    ax.minorticks_off()
    # ax.yaxis.set_ticks_position('both')
    # ax.xaxis.set_ticks_position('both')

    # Limits of graph
    ax.set_xlim([0, x_limit])
    ax.set_ylim([0, y_limit])
    
    # Labels    
    ax.set_xlabel('PRESIÓN DE POROS \n (kPa)') 
    ax.set_ylabel('PROF. (m)')  
    ax.invert_yaxis()
    
    # pp = PdfPages('tes2t.pdf')
    # pp.savefig(plt.gcf())
    
    time_now = datetime.now()
    current_time = time_now.strftime("%Y-%m-%d %H_%M")
    plt.savefig("Profile-pdf" + data[-1] + "_" + current_time + ".pdf")
    # plt.tight_layout()
    return 0

def pre_graph_Ic(data, y_limit, ax):
    '''
    This function create the profile of Ic vs Elevation
    data: Data provided of process_sbt_data()
    '''
    
    cpt_data = data[0]
    
    ax.plot(cpt_data["I_c"],
            cpt_data["z (m)"],
            "-",
            color = "black")
    
    # Creating a fill to see better the separation
    ax.fill_betweenx(
    cpt_data["z (m)"],
    0,
    cpt_data["I_c"],
    where = cpt_data["I_c"] <= 2.6, interpolate= True,
    facecolor="green",
    alpha = 0.8 )
    
    ax.fill_betweenx(
    cpt_data["z (m)"],
    0,
    cpt_data["I_c"],
    where = (cpt_data["I_c"] > 2.6) & (cpt_data["I_c"] < 3), interpolate=True,
    facecolor="blue",
    alpha = 0.8)
    
    ax.fill_betweenx(
    cpt_data["z (m)"],
    0,
    cpt_data["I_c"],
    where = cpt_data["I_c"] >= 3, interpolate=True,
    facecolor="red",
    alpha = 0.8)    
    
    #  Line wich separte the behaviour of Clay Like and Sand Like
    ax.plot(
        [2.6, 2.6], 
        [0, float(y_limit)],
        "--", 
        color = "salmon",
        label = "Ic = 2.6"
    )
    ax.plot(
        [3, 3], 
        [0, float(y_limit)],
        "--", 
        color = "salmon",
        label = "Ic = 2.6"
    )
    
    # Limits of graphs
    ax.set_xlim(1, 4)
    ax.set_ylim(0, float(y_limit))
    
    # Labels of graph
    ax.set_ylabel("Depth (m)")
    ax.set_xlabel('$I_C$') 
    ax.set_title("Soil behaviour type index ($I_C$)\n" + data[-1])

    # Ticks
    ax.tick_params(axis="x",
                direction="in",
                length=6)
    ax.tick_params(axis="y",
                direction="in",
                length=6)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(axis="y", which='minor',direction="in", length=3)
    ax.tick_params(axis="x", which='minor',direction="in", length=3)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    
    green_proxy = plt.Rectangle((0, 0), 1, 1, fc="green", alpha = 0.8)
    red_proxy = plt.Rectangle((0, 0), 1, 1, fc="red", alpha = 0.8)
    blue_proxy = plt.Rectangle((0, 0), 1, 1, fc="blue", alpha = 0.8)

    ax.legend([green_proxy, 
               blue_proxy,
               red_proxy], 
              [r'$I_c \leq 2.6$: Sand-like soils', 
               r'$2.6 < I_c < 3,$: Transition', 
               r'$I_c \geq 3$: Clay-like soils'],
              bbox_to_anchor = (0.5, -0.12),
              loc = "lower center",
              ncol = 1,
              edgecolor = "None",
              fontsize = 8
              )
    
    # Grid and Inversion of yaxis
    # ax.grid(True, 
    #         ls="-",
    #         color='0.9')
    
    ax.invert_yaxis()

    # time_now = datetime.now()
    # current_time = time_now.strftime("%Y-%m-%d %H_%M")
    # fig.savefig("prueba_" + data[-1] + "_" + current_time + ".svg")

    return 0

def pre_graph_IB(data, y_limit, ax):
    '''
    This function create the profile of IB vs Elevation
    data: Data provided of process_sbt_data()
    '''
    
    cpt_data = data[0]
    
    ax.plot(cpt_data["I_B"],
            cpt_data["z (m)"],
            "-",
            color = "black")
    
    # Creating a fill to see better the separation
    ax.fill_betweenx(
    cpt_data["z (m)"],
    0,
    cpt_data["I_B"],
    where = cpt_data["I_B"] <= 22, interpolate= True,
    facecolor="red",
    alpha = 0.8 )
    
    ax.fill_betweenx(
    cpt_data["z (m)"],
    0,
    cpt_data["I_B"],
    where = (cpt_data["I_B"] > 22) & (cpt_data["I_B"] < 32), interpolate=True,
    facecolor="blue",
    alpha = 0.8)
    
    ax.fill_betweenx(
    cpt_data["z (m)"],
    0,
    cpt_data["I_B"],
    where = cpt_data["I_B"] >= 32, interpolate=True,
    facecolor="green",
    alpha = 0.8)    
    
    #  Line wich separte the behaviour of Clay Like and Sand Like
    ax.plot(
        [22, 22], 
        [0, float(y_limit)],
        "--", 
        color = "salmon",
        label = "IB = 22"
    )
    ax.plot(
        [32, 32], 
        [0, float(y_limit)],
        "--", 
        color = "salmon",
        label = "IB = 32"
    )
    
    # Limits of graphs
    ax.set_xlim(10, 100)
    ax.set_ylim(0, float(y_limit))
    ax.set_xscale('log')
    
    # Labels of graph
    ax.set_ylabel("Depth (m)")
    ax.set_xlabel('$I_B$') 
    ax.set_title("Modified SBT index ($I_B$)\n" + data[-1])

    # Ticks
    ax.tick_params(axis="x",
                direction="in",
                length=6)
    ax.tick_params(axis="y",
                direction="in",
                length=6)
    # ax.xaxis.set_major_locator(ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    
    ax.tick_params(axis="y", which='minor',direction="in", length=3)
    ax.tick_params(axis="x", which='minor',direction="in", length=3)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    
    green_proxy = plt.Rectangle((0, 0), 1, 1, fc="green", alpha = 0.8)
    red_proxy = plt.Rectangle((0, 0), 1, 1, fc="red", alpha = 0.8)
    blue_proxy = plt.Rectangle((0, 0), 1, 1, fc="blue", alpha = 0.8)

    ax.legend([green_proxy, 
               blue_proxy,
               red_proxy], 
              [r'$I_B \geq 32$: Sand-like soils', 
               r'$22 < I_B < 32$: Transition', 
               r'$I_B \leq 22$: Clay-like soils'],
              bbox_to_anchor = (0.5, -0.12),
              loc = "lower center",
              ncol = 1,
              edgecolor = "None",
              fontsize = 8
              )
    
    # Grid and Inversion of yaxis
    # ax.grid(True, 
    #         ls="-",
    #         color='0.9')
    
    ax.invert_yaxis()

    # time_now = datetime.now()
    # current_time = time_now.strftime("%Y-%m-%d %H_%M")
    # fig.savefig("prueba_" + data[-1] + "_" + current_time + ".svg")

    return 0

    
def pre_graph_qt(data, y_limit, ax):
    
    '''
    Characteristics of the graph qt
    '''
    
    cpt_data = data[0]
    ax.plot(cpt_data["qt (MPa)"], cpt_data["z (m)"], "-", color = "blue")
    
    # Limits of graphs
    ax.set_ylim(0, float(y_limit))
    ax.set_xlim(0, max(cpt_data["qt (MPa)"]) * 1.2)
    
    # Labels of graph
    ax.set_ylabel("Depth (m)")
    ax.set_xlabel('$q_t$ (MPa) ') 
    ax.set_title("Corrected cone resistance ($q_t$)\n" + data[-1])

    # Ticks
    ax.tick_params(axis="x",
                direction="in",
                length=6)
    ax.tick_params(axis="y",
                direction="in",
                length=6)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(2.5))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(axis="y", which='minor',direction="in", length=3)
    ax.tick_params(axis="x", which='minor',direction="in", length=3)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')

    ax.grid(True, 
        ls="-",
        color='0.9')   
    
    ax.invert_yaxis()
    return 0

def pre_graph_Rf(data, y_limit, ax):
    
    '''
    Characteristics of the graph qt
    '''
    
    cpt_data = data[0]
    ax.plot(cpt_data["Rf (%)"], cpt_data["z (m)"], "-", color = "blue")
    
    # Limits of graphs
    ax.set_ylim(0, float(y_limit))
    ax.set_xlim(0, max(cpt_data["Rf (%)"]) * 1.2)
    
    # Labels of graph
    ax.set_ylabel("Profundidad (m)")
    ax.set_xlabel('$R_f$ (%) ') 
    ax.set_title("Friction ratio ($R_f$)\n" + data[-1])

    # Ticks
    ax.tick_params(axis="x",
                direction="in",
                length=6)
    ax.tick_params(axis="y",
                direction="in",
                length=6)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(axis="y", which='minor',direction="in", length=3)
    ax.tick_params(axis="x", which='minor',direction="in", length=3)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')

    ax.grid(True, 
        ls="-",
        color='0.9')   
    
    ax.invert_yaxis()
    return 0

def pre_graph_Bq(data, y_limit, ax):
    
    '''
    Characteristics of the graph qt
    '''
    
    cpt_data = data[0]
    ax.plot(cpt_data["B_q"], cpt_data["z (m)"], "-", color = "blue")
    
    # Limits of graphs
    ax.set_ylim(0, float(y_limit))
    ax.set_xlim(0, 1)
    # Labels of graph
    ax.set_ylabel("Profundidad (m)")
    ax.set_xlabel('$B_q$') 
    ax.set_title("Pore pressure ratio ($B_q$)\n" + data[-1])

    # Ticks
    ax.tick_params(axis="x",
                direction="in",
                length=6)
    ax.tick_params(axis="y",
                direction="in",
                length=6)
    
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    
    ax.tick_params(axis="y", which='minor',direction="in", length=3)
    ax.tick_params(axis="x", which='minor',direction="in", length=3)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')

    ax.grid(True, 
        ls="-",
        color='0.9')   
    
    ax.invert_yaxis()
    return 0


def pre_graph_Qtn(data, y_limit, ax):
    
    '''
    Characteristics of the graph Qtn
    '''
    
    cpt_data = data[0]
    ax.plot(cpt_data["Q_tn"], cpt_data["z (m)"], "-", color = "blue")
    
    # Limits of graphs
    ax.set_ylim(0, float(y_limit))
    ax.set_xlim(0, max(cpt_data["Q_tn"]) * 1.2)
    
    # Labels of graph
    ax.set_ylabel("Depth (m)")
    ax.set_xlabel('$Q_{tn}$ (kPa) ') 
    ax.set_title("Normalized cone resistance ($Q_{tn}$)\n" + data[-1])

    # Ticks
    ax.tick_params(axis="x",
                direction="in",
                length=6)
    ax.tick_params(axis="y",
                direction="in",
                length=6)
    # ax.xaxis.set_major_locator(MultipleLocator(50))
    # ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(axis="y", which='minor',direction="in", length=3)
    ax.tick_params(axis="x", which='minor',direction="in", length=3)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')

    ax.grid(True, 
        ls="-",
        color='0.9')   
    
    ax.invert_yaxis()
    return 0

  
def graph_basic_plots(data, y_limit):
    '''
    Create a plot of the first 5 results plots
    '''
    
    fig, axes = plt.subplots(nrows = 1,
                    ncols = 6,
                    figsize = (18, 12), sharey=True)
    
    
    pre_graph_qt(data, y_limit, axes[0])
    pre_graph_Rf(data, y_limit, axes[1])
    pre_graph_Bq(data, y_limit, axes[2])
    pre_graph_Qtn(data, y_limit, axes[3])
    pre_graph_Ic(data, y_limit, axes[4])
    pre_graph_IB(data, y_limit, axes[5])
    
    axes[1].set_ylabel("")
    axes[2].set_ylabel("")
    axes[3].set_ylabel("")
    axes[4].set_ylabel("")
    axes[5].set_ylabel("")
    
    plt.subplots_adjust(wspace=0.5)
    plt.show()
    
    # time_now = datetime.now()
    # current_time = time_now.strftime("%Y-%m-%d %H_%M")
    # data[0].to_excel("results" + data[-1] +"_" + current_time + ".xlsx", index = False)  
    
    # time_now = datetime.now()
    # current_time = time_now.strftime("%Y-%m-%d %H_%M")
    # fig.savefig("basic_plots_" + data[-1] + "_" + current_time + ".svg")
    
    return 0

def multiply_graph_qt(list_data, y_limit):
    """
    Create multiply qt plots in a line
    """
    n_graphs = len(list_data)
    
    fig, axes = plt.subplots(nrows = 1,
                             ncols = n_graphs,
                             figsize = (n_graphs * 3, 12), sharey = True)
    
    for i, _ in enumerate(axes):
        pre_graph_qt(list_data[i], y_limit, axes[i])
        if i != 0:
            axes[i].set_ylabel("")
        else:
            pass
    
    plt.subplots_adjust(wspace = 0.5)
    plt.show()
    
def multiply_graph_Qtn(list_data, y_limit):
    """
    Create multiply Qtn plots in a line
    """
    n_graphs = len(list_data)
    
    fig, axes = plt.subplots(nrows = 1,
                             ncols = n_graphs,
                             figsize = (n_graphs * 3, 12), sharey = True)
    
    for i, _ in enumerate(axes):
        pre_graph_Qtn(list_data[i], y_limit, axes[i])
        if i != 0:
            axes[i].set_ylabel("")
        else:
            pass
    
    plt.subplots_adjust(wspace = 0.5)
    plt.show()

def multiply_graph_Ic(list_data, y_limit):
    """
    Create multiply Ic plots in a line
    """
    n_graphs = len(list_data)
    
    fig, axes = plt.subplots(nrows = 1,
                             ncols = n_graphs,
                             figsize = (n_graphs * 3, 12), sharey = True)
    
    for i, _ in enumerate(axes):
        pre_graph_Ic(list_data[i], y_limit, axes[i])
        if i != 0:
            axes[i].set_ylabel("")
        else:
            pass
    
    plt.subplots_adjust(wspace = 0.5)
    plt.show()
    
def multiply_graph_IB(list_data, y_limit):
    """
    Create multiply IB plots in a line
    """
    n_graphs = len(list_data)
    
    fig, axes = plt.subplots(nrows = 1,
                             ncols = n_graphs,
                             figsize = (n_graphs * 3, 12), sharey = True)
    
    for i, _ in enumerate(axes):
        pre_graph_IB(list_data[i], y_limit, axes[i])
        if i != 0:
            axes[i].set_ylabel("")
        else:
            pass
    
    plt.subplots_adjust(wspace = 0.5)
    plt.show()

def combine_cpt_vel(data, initial_value = 0):
    '''
    This function interpolate the data of Vs and add it in the cpt data
    data: Data provided of process_sbt_data()
    '''   
    cpt_data = data[0]
    vel_data = data[8]
    
    cpt_data = pd.merge(cpt_data,
                        vel_data, 
                        on = "z (m)", 
                        how = "outer")

    cpt_data = cpt_data.sort_values(by = "z (m)")
    cpt_data.reset_index(drop=True,inplace=True)

    cpt_data.set_index('z (m)',
                   inplace = True)

    cpt_data.interpolate(method = 'index',
                        inplace = True,
                        limit_direction = 'forward')

    cpt_data.reset_index(inplace = True)
    
    if initial_value == 0:
        cpt_data.loc[cpt_data["z (m)"] < vel_data["z (m)"].iloc[0], "Vs (m/s)"] = vel_data["Vs (m/s)"][0]
    else:
        cpt_data.loc[cpt_data["z (m)"] < vel_data["z (m)"].iloc[0], "Vs (m/s)"] = initial_value
    
    
    
    
    data[0] = cpt_data
    return data
        
             
def pre_graph_Vs(data, y_limit, ax):
    '''
    This function create the profile of K*c 
    data: Data provided of process_sbt_data()
    '''

    
    cpt_data = data[0]
    vel_data = data[8]
      
    ax.plot(cpt_data["Vs (m/s)"], cpt_data["z (m)"], "-", color = "blue")
    ax.plot(vel_data["Vs (m/s)"], vel_data["z (m)"], "s", color = "darkblue", markersize = 4) 
    # Limits of graphs
    ax.set_ylim(0, float(y_limit))
    ax.set_xlim(0, max(cpt_data["Vs (m/s)"]) * 1.2)
    
    # Labels of graph
    ax.set_ylabel("Profundidad (m)")
    ax.set_xlabel('$V_s$ (m/s) ') 
    ax.set_title("Shear Velocities ($V_s$)\n" + data[-1])

    # Ticks
    ax.tick_params(axis="x",
                direction="in",
                length=6)
    ax.tick_params(axis="y",
                direction="in",
                length=6)
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.xaxis.set_minor_locator(MultipleLocator(25))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(axis="y", which='minor',direction="in", length=3)
    ax.tick_params(axis="x", which='minor',direction="in", length=3)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')

    ax.grid(True, 
        ls="-",
        color='0.9')   
    
    ax.invert_yaxis()
    return 0

def graph_vs_plots(data, y_limit):
    
    fig, axes = plt.subplots(nrows = 1,
                ncols = 1,
                figsize = (3, 12))
    
    pre_graph_Vs(data, y_limit, axes)
    plt.show()
    return 0
    
def pre_graph_COV_Qtn(data, soil_type,  y_limit, ax):
    """
    Obtaining the Coeficient of Variation 
    """
    
    # La porción
    cpt_data = data[0].iloc[81:481]
    
    # print(cpt_data["I_c"]<=2.6)
    
    # La arena
    if soil_type == "Sand":
        cpt_data = cpt_data.loc[cpt_data["I_c"] <= 2.6]
    
    
    Q_tn_mean = cpt_data["Q_tn"].mean()
    Q_tn_std = cpt_data["Q_tn"].std()
    
    Q_tn_COV = Q_tn_std * 100 / Q_tn_mean
    

    
    ax.plot(cpt_data["Q_tn"], cpt_data["z (m)"], "o", color = "blue")
    
    # Limits of graphs
    ax.set_ylim(0, float(y_limit))
    ax.set_xlim(0, max(cpt_data["Q_tn"]) * 1.2)
    
    # Labels of graph
    ax.set_ylabel("Depth (m)")
    ax.set_xlabel('$Q_{tn}$ (kPa) ') 
    ax.set_title("Normalized cone resistance ($Q_{tn}$)\n" + data[-1] + "\n COV =" + str(round(Q_tn_COV, 2)))

    # Ticks
    ax.tick_params(axis="x",
                direction="in",
                length=6)
    ax.tick_params(axis="y",
                direction="in",
                length=6)


    ax.plot(
        [Q_tn_mean, Q_tn_mean], 
        [0, float(y_limit)],
        "--", 
        color = "salmon",
        label = "Mean $Q_{tn}$"
    )
    
    ax.plot(
        [Q_tn_mean + Q_tn_std, Q_tn_mean + Q_tn_std], 
        [0, float(y_limit)],
        "--", 
        color = "blue",
        label = "Standard Desviation $Q_{tn}$"
    )    

    ax.plot(
        [Q_tn_mean - Q_tn_std, Q_tn_mean - Q_tn_std], 
        [0, float(y_limit)],
        "--", 
        color = "blue"
    )       
    
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(axis="y", which='minor',direction="in", length=3)
    ax.tick_params(axis="x", which='minor',direction="in", length=3)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')

    ax.grid(True, 
        ls="-",
        color='0.9')   
    
    ax.legend(bbox_to_anchor = (0.5, -0.12),
              loc = "lower center",
              ncol = 1,
              edgecolor = "None",
              fontsize = 8
              )
    
    ax.invert_yaxis()
    return 0

def pre_graph_COV_qt(data, soil_type,  y_limit, ax):
    """
    Obtaining the Coeficient of Variation 
    """
    
    # La porción
    cpt_data = data[0].iloc[81:481]
    
    # print(cpt_data["I_c"]<=2.6)
    
    # La arena
    if soil_type == "Sand":
        cpt_data = cpt_data.loc[cpt_data["I_c"] <= 2.6]
    
    
    q_t_mean = cpt_data["qt (MPa)"].mean()
    q_t_std = cpt_data["qt (MPa)"].std()
    
    q_t_COV = q_t_std * 100 / q_t_mean
    

    
    ax.plot(cpt_data["qt (MPa)"], cpt_data["z (m)"], "o", color = "blue")
    
    # Limits of graphs
    ax.set_ylim(0, float(y_limit))
    ax.set_xlim(0, max(cpt_data["qt (MPa)"]) * 1.2)
    
    # Labels of graph
    ax.set_ylabel("Depth (m)")
    ax.set_xlabel('$q_{t}$ (kPa) ') 
    ax.set_title("Corrected cone resistance ($q_{t}$)\n" + data[-1] + "\n COV =" + str(round(q_t_COV, 2)))

    # Ticks
    ax.tick_params(axis="x",
                direction="in",
                length=6)
    ax.tick_params(axis="y",
                direction="in",
                length=6)


    ax.plot(
        [q_t_mean, q_t_mean], 
        [0, float(y_limit)],
        "--", 
        color = "salmon",
        label = "Mean $q_{t}$"
    )
    
    ax.plot(
        [q_t_mean + q_t_std, q_t_mean + q_t_std], 
        [0, float(y_limit)],
        "--", 
        color = "blue",
        label = "Standard Desviation $q_{t}$"
    )    

    ax.plot(
        [q_t_mean - q_t_std, q_t_mean - q_t_std], 
        [0, float(y_limit)],
        "--", 
        color = "blue"
    )       
    
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(axis="y", which='minor',direction="in", length=3)
    ax.tick_params(axis="x", which='minor',direction="in", length=3)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')

    ax.grid(True, 
        ls="-",
        color='0.9')   
    
    ax.legend(bbox_to_anchor = (0.5, -0.12),
              loc = "lower center",
              ncol = 1,
              edgecolor = "None",
              fontsize = 8
              )
    
    ax.invert_yaxis()
    return 0

def multiply_graph_COV_Qtn(list_data, soil_type, y_limit):
    """
    Create multiply COV Qtn plots in a line
    """
    n_graphs = len(list_data)
    
    fig, axes = plt.subplots(nrows = 1,
                             ncols = n_graphs,
                             figsize = (n_graphs * 3, 12), sharey = True)
    
    for i, _ in enumerate(axes):
        pre_graph_COV_Qtn(list_data[i], soil_type, y_limit, axes[i])
        if i != 0:
            axes[i].set_ylabel("")
        else:
            pass
    
    plt.subplots_adjust(wspace = 0.5)
    plt.show()
    
def multiply_graph_COV_qt(list_data, soil_type, y_limit):
    """
    Create multiply COV Qtn plots in a line
    """
    n_graphs = len(list_data)
    
    fig, axes = plt.subplots(nrows = 1,
                             ncols = n_graphs,
                             figsize = (n_graphs * 3, 12), sharey = True)
    
    for i, _ in enumerate(axes):
        pre_graph_COV_qt(list_data[i], soil_type, y_limit, axes[i])
        if i != 0:
            axes[i].set_ylabel("")
        else:
            pass
    
    plt.subplots_adjust(wspace = 0.5)
    plt.show()