import os
import torch
from datetime import datetime as dt
from threading import Thread
import PySimpleGUI as sg
from pruneforui import prune_it
from weight_matching import sdunet_permutation_spec, weight_matching, apply_permutation
from util.ui_flattener import flatten_ui_elements
import util.progress_bar_custom as cpbar
import util.colors as color
import util.support as support
from CONSTANTS import *

__version__ = '0.0.5'
sg.theme('Dark Gray 15')
APP_TITLE = f"Merge-Stable-Diffusion-models-without-distortion-GUI - Ver {__version__}"

def main():
    start_time = dt.now()
    lower_iter_spinbox, upper_iter_spinbox = 1, 100; data_iter_spinbox = [i for i in range(lower_iter_spinbox - 1, upper_iter_spinbox + 2)]
    #region layout
    top_column = [
        [
            support.buttons_layout(),
        ],     
            # spacer
            [sg.T("")],
        [
        sg.Frame('',[
                [
                    sg.Input(key=MODEL_A_INP_KEY,enable_events=True,expand_x=True,expand_y=True,background_color=color.DARK_GRAY),
                    sg.FileBrowse(MODEL_A_LBL,file_types=FILE_EXT,size=(12,1)),                     
                ],
                [
                    sg.Input(key=MODEL_B_INP_KEY,enable_events=True,expand_x=True,expand_y=True,background_color=color.DARK_GRAY),
                    sg.FileBrowse(MODEL_B_LBL,file_types=FILE_EXT,size=(12,1))                            
                ],
                [
                    sg.Input(key=MERGED_MODEL_INP_KEY,enable_events=True,expand_x=True,expand_y=True,background_color=color.DARK_GRAY),
                    sg.FileSaveAs(MERGED_MODEL_LBL,file_types=FILE_EXT,size=(12,1))                            
                ],                     
                [
                    sg.Combo(DEVICE,default_value=DEVICE[0],key=SELECTED_DEVICE_COMBO_KEY,readonly=True,text_color=color.TERMINAL_BLUE,background_color=color.GRAY_9900,visible=False),
                    sg.Checkbox(USE_FP16_LBL,k=USE_FP16_CHKBOX_KEY,default=True,enable_events=True,font=FONT),
                    sg.T(ALPHA_LBL,font=FONT),
                    sg.Slider(default_value=0.5,range=((0.01,1)),resolution=0.01,    
                    orientation='horizontal',disable_number_display=True,enable_events=True,k=ALPHA_SLDR_KEY,expand_x=True,s=(12,12)),   
                    sg.In(0.5,k=ALPHA_SLDR_INP_KEY,s=(5,5),justification='center',enable_events=True,readonly=True,disabled_readonly_background_color=color.GRAY_1111,font=FONT),
                    sg.T(ITERATIONS_LBL,font=FONT),
                    sg.Spin(data_iter_spinbox, initial_value=1, size=3, enable_events=True, key=ITERATIONS_SPIN_BOX_KEY,font=FONT,),
                ],
        ],expand_x=True,relief=sg.RELIEF_SOLID,border_width=1,background_color=color.GRAY_9900,element_justification="c")            
     ],
    ]

    console_column = [
        [
            sg.Frame('',[       
                    [
                        sg.Button(MERGE_MODELS_LBL,k=MERGE_MODELS_BTN_KEY,font=FONT,expand_x=True,size=(30,2),mouseover_colors=(color.GRAY_9900,color.DARK_GREEN)),
                        sg.Button(PRUNE_MODEL_LBL,k=PRUNE_MODEL_BTN_KEY,font=FONT,expand_x=True,size=(30,2),mouseover_colors=(color.GRAY_9900,color.DARK_GREEN)),
                    ],                     
                    [    
                        sg.Multiline(GREET_MSG,k=CONSOLE_ML_KEY,visible=True,text_color=color.TERMINAL_GREEN,background_color=color.GRAY_1111,
                        reroute_stdout=True,write_only=False,reroute_cprint=True, autoscroll=True, auto_refresh=True,size=(100,20),expand_x=True,expand_y=True,font=FONT),
                    ], 
            ],expand_x=True,expand_y=True,border_width=0,relief=sg.RELIEF_FLAT,element_justification="c",background_color=color.GRAY_9900)
        ],  
    ]
  
    bottom_column = [
        [
            sg.Frame('',[                                    
                    [
                        cpbar.progress_bar_custom_layout(PBAR_KEY,visible=True)
                    ],
            ],expand_x=True,border_width=0,relief=sg.RELIEF_FLAT,element_justification='c')
        ],    
    ]

    layout = [[
             top_column,       
            [
                sg.Column(console_column,element_justification='r', expand_x=True,expand_y=True,visible=True),
            ],        
             bottom_column,
        ]
    ]


    #endregion layout

    window = sg.Window(APP_TITLE,layout,finalize=True, resizable=True,enable_close_attempted_event=False,background_color=color.GRAY_9900)

    console_ml_elem:sg.Multiline = window[CONSOLE_ML_KEY]
    merged_model_inp_elem:sg.Input = window[MERGED_MODEL_INP_KEY]
    alpha_inp_elem:sg.Input = window[ALPHA_SLDR_INP_KEY]
    merge_models_btn_elem:sg.Button = window[MERGE_MODELS_BTN_KEY]
    prune_model_btn_elem:sg.Button = window[PRUNE_MODEL_BTN_KEY]
    iterations_spin_elem:sg.Spin = window[ITERATIONS_SPIN_BOX_KEY]

    flatten_ui_elements(window)

    def merge_models_thread(model_a:str, model_b:str, device:str="cpu", output:str="merged", usefp16:bool=True, alpha:float=0.5, iterations:int=10):
        
        def flatten_params(model):
            return model["state_dict"]

        def merge_models(model_a:str, model_b:str, device:str="cpu", output:str="merged", usefp16:bool=True, alpha:float=0.5, iterations:int=10):
            print(f"""
            ---------------------
                model_a:    {os.path.basename(model_a)}
                model_b:    {os.path.basename(model_b)}
                output:     {os.path.basename(output)}
                alpha:      {alpha}
                usefp16:    {usefp16}  
                iterations: {iterations}
            ---------------------
            """)        
            model_a = torch.load(model_a, map_location=device)
            model_b = torch.load(model_b, map_location=device)
            theta_0 = model_a["state_dict"]
            theta_1 = model_b["state_dict"]

            alpha = float(alpha)
            iterations = int(iterations)
            step = alpha/iterations
            permutation_spec = sdunet_permutation_spec()
            special_keys = ["first_stage_model.decoder.norm_out.weight", "first_stage_model.decoder.norm_out.bias", "first_stage_model.encoder.norm_out.weight", 
            "first_stage_model.encoder.norm_out.bias", "model.diffusion_model.out.0.weight", "model.diffusion_model.out.0.bias"]
            if usefp16:
                print("Using half precision")
            else:
                print("Using full precision")

            for x in range(iterations):
                print(f"""
                ---------------------
                    ITERATION {x+1}
                ---------------------
                """)

                # In order to reach a certain alpha value with a given number of steps,
                # You have to calculate an alpha for each individual iteration
                if x > 0:
                    new_alpha = 1 - (1 - step*(1+x)) / (1 - step*(x))
                else:
                    new_alpha = step
                print(f"new alpha = {new_alpha}\n")


                theta_0 = {key: (1 - (new_alpha)) * theta_0[key] + (new_alpha) * value for key, value in theta_1.items() if "model" in key and key in theta_1}

                if x == 0:
                    for key in theta_1.keys():
                        if "model" in key and key not in theta_0:
                            theta_0[key] = theta_1[key]

                print("FINDING PERMUTATIONS")

                # Replace theta_0 with a permutated version using model A and B    
                first_permutation, y = weight_matching(permutation_spec, flatten_params(model_a), theta_0, usefp16=usefp16)
                theta_0 = apply_permutation(permutation_spec, first_permutation, theta_0)
                second_permutation, z = weight_matching(permutation_spec, flatten_params(model_b), theta_0, usefp16=usefp16)
                theta_3= apply_permutation(permutation_spec, second_permutation, theta_0)

                new_alpha = torch.nn.functional.normalize(torch.sigmoid(torch.Tensor([y, z])), p=1, dim=0).tolist()[0]

                # Weighted sum of the permutations
                
                for key in special_keys:
                    theta_0[key] = (1 - new_alpha) * (theta_0[key]) + (new_alpha) * (theta_3[key])

                cpbar.progress_bar_custom(x,iterations,start_time,window,PBAR_KEY)

            output_file = f'{output}'
    
            print(f"\nSaving... \n\n{output_file}")

            torch.save({
                    "state_dict": theta_0
                        }, output_file)

            print("\nDone!")    

            merge_bt_enable()
        
        try:
            merge_models(model_a=model_a, model_b=model_b, device=device, output=output, usefp16=usefp16, alpha=alpha, iterations=iterations)   
        except RuntimeError as e:
            print(RuntimeError,e)
            merge_bt_enable()
            cpbar.progress_bar_reset(window,PBAR_KEY)
        except KeyError as e:
            print(KeyError,e)
            merge_bt_enable()
            cpbar.progress_bar_reset(window,PBAR_KEY)

    #region disable/enable buttons

    def merge_bt_disable():
        merge_models_btn_elem.update(MERGING_MODELS_LBL)
        merge_models_btn_elem.update(disabled_button_color=(color.DARK_GREEN,color.GRAY_9900))
        merge_models_btn_elem.update(disabled=True) 
        prune_model_btn_elem.update(disabled_button_color=(color.DARK_GREEN,color.GRAY_9900))
        prune_model_btn_elem.update(disabled=True)
   
    def merge_bt_enable():
        merge_models_btn_elem.update(disabled=False)
        merge_models_btn_elem.update(MERGE_MODELS_LBL) 
        prune_model_btn_elem.update(disabled=False)

    def prune_bt_disable():
        merge_models_btn_elem.update(disabled_button_color=(color.DARK_GREEN,color.GRAY_9900))
        merge_models_btn_elem.update(disabled=True) 
        prune_model_btn_elem.update(PRUNING_MODEL_LBL)
        prune_model_btn_elem.update(disabled_button_color=(color.DARK_GREEN,color.GRAY_9900))
        prune_model_btn_elem.update(disabled=True)
   
    def prune_bt_enable():
        merge_models_btn_elem.update(disabled=False)
        prune_model_btn_elem.update(disabled=False)
        prune_model_btn_elem.update(PRUNE_MODEL_LBL)

    #endregion

    def merge_modelname(model_a:str, model_b:str, usefp16:bool=True, alpha:float=0.1, iterations:int=10)-> str:
        alpha_b = 1 - (float(alpha))
        alpha_b = round(alpha_b, 2)
        fp16 = "_fp16" if usefp16 else ""
        output = f"{os.path.dirname(model_a)}/{os.path.splitext(os.path.basename(model_a))[0]}_{alpha}_{os.path.splitext(os.path.basename(model_b))[0]}_{alpha_b}_{iterations}it{fp16}.ckpt"
        return output

    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED:
            break
  
        if event == ALPHA_SLDR_KEY:
            alpha_inp_elem.update(values[ALPHA_SLDR_KEY])

        if event == MERGE_MODELS_BTN_KEY:
            model_a = values[MODEL_A_INP_KEY]
            model_b = values[MODEL_B_INP_KEY]
            output = values[MERGED_MODEL_INP_KEY]
            device = values[SELECTED_DEVICE_COMBO_KEY]
            usefp16 = values[USE_FP16_CHKBOX_KEY]
            alpha = values[ALPHA_SLDR_INP_KEY]
            iterations = values[ITERATIONS_SPIN_BOX_KEY]
            cpbar.progress_bar_reset(window,PBAR_KEY)      

            if model_a and model_b and output:
                start_time = dt.today().timestamp()
                cpbar.progress_bar_calc(window,PBAR_KEY)
                merge_bt_disable()    
                console_ml_elem.update("")        
                Thread(target=merge_models_thread, args=(model_a, model_b, device, output, usefp16, alpha, iterations), daemon=True).start()    
            else:
                print(MISSING_MODEL_PATH_TXT)

        if event == PRUNE_MODEL_BTN_KEY:
            model_a = values[MODEL_A_INP_KEY]
      
            if model_a:
                start_time = dt.today().timestamp()
                cpbar.progress_bar_reset(window,PBAR_KEY)      
                prune_bt_disable()
        
                console_ml_elem.update("")        
                try:
                    cpbar.progress_bar_calc(window,PBAR_KEY)
                    prune_it(model_a)
                    cpbar.progress_bar_custom(0,1,start_time,window,PBAR_KEY)
                    prune_bt_enable()

                except KeyError as e:
                    print(KeyError,e)
                    prune_bt_enable()
                    cpbar.progress_bar_reset(window,PBAR_KEY)
            else:
                print(MISSING_MODEL_PATH_TXT)

        if event in (ALPHA_SLDR_KEY, USE_FP16_CHKBOX_KEY,ITERATIONS_SPIN_BOX_KEY,MODEL_A_INP_KEY, MODEL_B_INP_KEY):
            if values[MODEL_A_INP_KEY] and values[MODEL_B_INP_KEY]:
                merged_model_inp_elem.update(merge_modelname(values[MODEL_A_INP_KEY], values[MODEL_B_INP_KEY], usefp16=values[USE_FP16_CHKBOX_KEY], alpha=values[ALPHA_SLDR_KEY], iterations=values[ITERATIONS_SPIN_BOX_KEY]))
       
        if event == ITERATIONS_SPIN_BOX_KEY:
            value = values[ITERATIONS_SPIN_BOX_KEY]
            if value == lower_iter_spinbox - 1:
                iterations_spin_elem.update(value=upper_iter_spinbox)
                values[ITERATIONS_SPIN_BOX_KEY] = upper_iter_spinbox
            elif value == upper_iter_spinbox + 1:
                iterations_spin_elem.update(value=lower_iter_spinbox)
                values[ITERATIONS_SPIN_BOX_KEY] = lower_iter_spinbox   
        
        support.buttons(event)
  
if __name__ == '__main__':

    GREET_MSG=f"""
        How to use:

            Merging models:

                1. Select Model A, Model B paths.
                2. Then enter the desired parameters for the merge, a new name will be generated.
                    Or you can enter a custom name.
                3. Then click the "MERGE MODELS" button.

            Pruning model:

                1. Select Model A path.
                2. Then enter the desired parameters for the prune.
                3. Then click the "PRUNE MODEL" button.

        Please consider donating to the project if you find it useful,
        so that I can maintain and improve this tool and other projects.
        """    

    main() 


