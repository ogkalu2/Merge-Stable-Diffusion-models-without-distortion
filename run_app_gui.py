import PySimpleGUI as sg
import torch 
import webbrowser
from threading import Thread
from datetime import datetime as dt
import util.icons as ic
import util.progress_bar_custom as cpbar
from weight_matching import sdunet_permutation_spec, weight_matching, apply_permutation
from pruneforui import prune_it

__version__ = '0.0.1'
APP_TITLE = f"Merge-Stable-Diffusion-models-without-distortion-GUI - Ver {__version__}"
sg.theme('Dark Gray 15')

#region constants
COLOR_DARK_GREEN = '#78BA04'
COLOR_GRAY_9900 = '#0A0A0A'
COLOR_DARK_GRAY = '#1F1F1F'
COLOR_BLUE_TERMINAL = '#0099ff'
COLOR_GRAY_1111 = '#111111'
PBAR_KEY = 'progress_bar'
#endregion

file_ext = {("All", "*.ckpt"),}
 
def main():
    start_time = dt.today().timestamp()

    #region layout
    top_column = [
        [
        sg.Frame('',[
                [
                    sg.Button(image_data=ic.patreon,key="-patreon-",button_color=(COLOR_GRAY_9900,COLOR_GRAY_9900)),
                    sg.Button(image_data=ic.supportme,visible=False,key="-supportme-",button_color=(COLOR_GRAY_9900,COLOR_GRAY_9900)),
                    sg.Button(image_data=ic.github,key="-github-",button_color=(COLOR_GRAY_9900,COLOR_GRAY_9900))
                ],
            ],expand_x=True,relief=sg.RELIEF_SOLID,border_width=1,background_color=COLOR_GRAY_9900,element_justification="r")
        ],     
        [
            sg.Push(),
        ],
        [
        sg.Frame('',[
                [
                    sg.Input(key=f'-model_a_input-',enable_events=True,expand_x=True,expand_y=True,background_color=COLOR_DARK_GRAY),
                    sg.FileBrowse("Model A",k=f'-model_a_FileBrowse-',file_types=(file_ext),size=(12,1)),                     
                ],
                [
                    sg.Input(key=f'-model_b_input-',enable_events=True,expand_x=True,expand_y=True,background_color=COLOR_DARK_GRAY),
                    sg.FileBrowse("Model B",k=f'-model_b_FileBrowse-',file_types=(file_ext),size=(12,1))                            
                ],
                [
                    sg.Input(key=f'-merged_model_in-',enable_events=True,expand_x=True,expand_y=True,background_color=COLOR_DARK_GRAY),
                    sg.FileSaveAs("Merged Model",k=f'-merged_model_FileSaveAs-',file_types=(file_ext),size=(12,1))                            
                ],                     
                [
                    sg.Combo(['cpu','gpu'],default_value='cpu',key='-selected_device-',readonly=True,text_color=COLOR_BLUE_TERMINAL,background_color=COLOR_GRAY_9900,visible=False),
                    sg.Checkbox('usefp16',k='-usefp16_checkbox-',default=True),
                    sg.T('alpha:'),
                    sg.Slider(default_value=0.5,range=((0.01,1)),resolution=0.01,    
                    orientation='horizontal',disable_number_display=True,enable_events=True,k='-alpha_slider-',expand_x=True,s=(12,12)),   
                    sg.In(0.5,k='-alpha_in-',s=(5,5),justification='center'),
                    sg.T('iterations:',),
                    sg.In(10,k='-iterations_in-',s=(5,5),justification='center'),
                ],
        ],expand_x=True,relief=sg.RELIEF_SOLID,border_width=1,background_color=COLOR_GRAY_9900)            
     ],
    ]

    console_column = [
        [
            sg.Frame('',[       
                    [
                        sg.Button('MERGE MODELS',k='-merge_models_bt-',font='Ariel 12 ',expand_x=True,size=(30,2),mouseover_colors=(COLOR_GRAY_9900,COLOR_DARK_GREEN)),
                        sg.Button('PRUNE MODEL',k='-prune_model_bt-',font='Ariel 12 ',expand_x=True,size=(30,2),mouseover_colors=(COLOR_GRAY_9900,COLOR_DARK_GREEN)),
                        sg.Checkbox('keep_only_ema',k='-keep_only_ema_checkbox-',default=False),
                    ],                     
                    [    
                        sg.MLine(GREET_MSG,k='-console_ml-',visible=True,text_color='#00cc00',background_color=COLOR_GRAY_1111,border_width=0,sbar_width=20,sbar_trough_color=0,
                        reroute_stdout=True,write_only=False,reroute_cprint=True, autoscroll=True, auto_refresh=True,size=(80,20),expand_x=True,expand_y=True,font="Ariel 11 "),
                    ], 
            ],expand_x=True,expand_y=True,border_width=0,relief=sg.RELIEF_FLAT,element_justification="c",background_color=COLOR_GRAY_9900)
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
                sg.Column(console_column, key='-console_column-', element_justification='r', expand_x=True,expand_y=True,visible=True),
            ],        
             bottom_column,
        ]
    ]

    #endregion layout

    window = sg.Window(APP_TITLE,layout,finalize=True, resizable=True,enable_close_attempted_event=False,background_color=COLOR_GRAY_9900)
    window.hide    

    #region widget and flating

    console_ml_widget = window["-console_ml-"] 
    patreon_widget = window["-patreon-"]
    supportme_widget = window["-supportme-"]
    github_widget = window["-github-"]
    model_a_input_widget = window["-model_a_input-"]
    model_b_input_widget = window["-model_b_input-"]
    merged_model_in_widget = window["-merged_model_in-"]
    model_a_bt_widget = window["-model_a_FileBrowse-"]
    model_b_bt_widget = window["-model_b_FileBrowse-"]
    merged_model_file_save_as_widget = window["-merged_model_FileSaveAs-"]
    alpha_in_widget = window["-alpha_in-"]
    iterations_in_widget = window["-iterations_in-"]
    merge_models_bt_widget = window["-merge_models_bt-"]
    prune_model_bt_widget = window["-prune_model_bt-"]


    widgets = {
        patreon_widget,
        github_widget,
        supportme_widget,
        console_ml_widget,
        model_a_bt_widget,
        model_b_bt_widget,
        merge_models_bt_widget,
        model_a_input_widget,
        model_b_input_widget,
        merged_model_in_widget,
        alpha_in_widget,
        iterations_in_widget,
        merged_model_file_save_as_widget,
        prune_model_bt_widget
    }

    for widget in widgets:
        widget.Widget.config(relief='flat')  

    #endregion 

    def flatten_params(model):
        return model["state_dict"]

    def merge_models(model_a, model_b, device="cpu", output="merged", usefp16=True, alpha="0.5", iterations="10"):
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
            cpbar.progress_bar_custom(x,iterations,start_time,window,PBAR_KEY)

            # Weighted sum of the permutations
            
            for key in special_keys:
                theta_0[key] = (1 - new_alpha) * (theta_0[key]) + (new_alpha) * (theta_3[key])

        output_file = f'{output}'

        print("\nSaving...")

        torch.save({
                "state_dict": theta_0
                    }, output_file)

        print("Done!")    

        merge_bt_enable()

    def merge_models_t(model_a, model_b, device="cpu", output="merged", usefp16=True, alpha="0.5", iterations="10"):
        
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

    def merge_bt_disable():
        merge_models_bt_widget.update('MERGING MODELS...')
        merge_models_bt_widget.update(disabled_button_color=(COLOR_DARK_GREEN,COLOR_GRAY_9900))
        merge_models_bt_widget.update(disabled=True) 
        prune_model_bt_widget.update(disabled_button_color=(COLOR_DARK_GREEN,COLOR_GRAY_9900))
        prune_model_bt_widget.update(disabled=True)
   
    def merge_bt_enable():
        merge_models_bt_widget.update(disabled=False)
        merge_models_bt_widget.update('MERGE MODELS') 
        prune_model_bt_widget.update(disabled=False)

    def prune_bt_disable():
        merge_models_bt_widget.update(disabled_button_color=(COLOR_DARK_GREEN,COLOR_GRAY_9900))
        merge_models_bt_widget.update(disabled=True) 
        prune_model_bt_widget.update('PRUNING MODEL...')
        prune_model_bt_widget.update(disabled_button_color=(COLOR_DARK_GREEN,COLOR_GRAY_9900))
        prune_model_bt_widget.update(disabled=True)
   
    def prune_bt_enable():
        merge_models_bt_widget.update(disabled=False)
        prune_model_bt_widget.update(disabled=False)
        prune_model_bt_widget.update('PRUNE MODEL')

    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED:
            break
  
        if event == "-alpha_slider-":
            alpha_in_widget.update(values["-alpha_slider-"])

        if event == "-merge_models_bt-":
            model_a = values["-model_a_input-"]
            model_b = values["-model_b_input-"]
            output = values["-merged_model_in-"]
            device = values["-selected_device-"]
            usefp16 = values["-usefp16_checkbox-"]
            alpha = values["-alpha_in-"]
            iterations = values["-iterations_in-"]
            cpbar.progress_bar_reset(window,PBAR_KEY)      

            if model_a and model_b and output:
                start_time = dt.today().timestamp()
                cpbar.progress_bar_calc(window,PBAR_KEY)
                merge_bt_disable()               
                console_ml_widget.update("")        
                Thread(target=merge_models_t, args=(model_a, model_b, device, output, usefp16, alpha, iterations), daemon=True).start()    
            else:
                print("missing model path")

        if event == "-prune_model_bt-":
            model_a = values["-model_a_input-"]
            keep_only_ema = values["-keep_only_ema_checkbox-"]
      
            if model_a:
                start_time = dt.today().timestamp()
                cpbar.progress_bar_reset(window,PBAR_KEY)      
                prune_bt_disable()
        
                console_ml_widget.update("")        
                try:
                    cpbar.progress_bar_calc(window,PBAR_KEY)
                    prune_it(model_a,keep_only_ema)
                    cpbar.progress_bar_custom(0,1,start_time,window,PBAR_KEY)
                    prune_bt_enable()

                except KeyError as e:
                    print(KeyError,e)
                    prune_bt_enable()
                    cpbar.progress_bar_reset(window,PBAR_KEY)
            else:
                print("missing model path")

        if event == "-patreon-":
            webbrowser.open("https://www.patreon.com/distyx")      
        if event == "-github-":
            webbrowser.open("https://github.com/diStyApps/Merge-Stable-Diffusion-models-without-distortion-gui")  
        if event == "-supportme-":
            webbrowser.open("https://coindrop.to/disty")  
  
if __name__ == '__main__':

    GREET_MSG=f"""
        Instructions:

        Merging models:

            1. Select Model A, Model B paths, Then select the Merged Model output path.
            2. Then enter the desired prameters for the merge.
            3. Then click the merge models button.


        Pruning model:

            1. Select Model A path.
            2. Then enter the desired parameters for the prune.
            3. Then click the prune model button.

        Please consider donating to the project if you find it useful,
        so that I can maintain and improve this tool and other projects.
        """    
 
    main() 


