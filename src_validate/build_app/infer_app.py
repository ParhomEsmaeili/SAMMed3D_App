
#Sample application adapted from the SAMMed3D repository's branch for the the SegFM challenge. 



# RAS: We provided image patch, and prompts generated in an RAS coordinate system, so we will not require any re-orientation.




#The SAMMed3D implementation reads in all the prompts in the implementation... but they only perform actions on the 
# foreground. This is because the logic that they use downstream would break if attempting to use it
# on the background, e.g.:

# Their prompt simulation (within the prompt region) would not work. Because they didn't do an actual mapping of the point
# coordinate to the model domain, as they went through other morphological operations (like resampling etc) and so used a 
# box mask which they would propagate through to the model domain and then sample a point from that. But they had not provided this for the background clicks.
# 
# Also, when they try to re-insert the changed voxels, they use an assignment according to semantic code. This also wouldn't work. 
# This would fail if background, as it typically is, has code of 0.  
# NOTE: Adjustments on our end could have been made to account for this somehow, but given that the foreground and background clicks may not necessary fall
# within the same resampled region, it would be non-trivial to implement this without changing their logic too much.

#The logic of their code is also written in a way that more than one point could not be passed through at the same time.
#The mechanism that they used was heavily dependent on the simulation strategy of the segFM challenge which presumed that the prompts
# were always sampled according to the maximal error region.
# 
#   - When they read in the clicks, they first append all the foreground, and then all the background for a given semantic class.
#   - IN THAT ORDER. 
#   
#   Then, when they are assessing whether or not to actually perform inference, they look at the semantic code of the final appended
#   click, and if it is a foreground code, they perform inference. But clearly, this will effectively terminate all refinement once
#   a background click is appended, as the last click will always be a background click. 

#   - This is because they do not have a mechanism to determine whether the last click was a foreground or background click,


#This is not going to work for us, as we want to be able to pass both foreground and background clicks at the same time. 

#So we will just skip over the background clicks, and if there are no foreground clicks we just have to return the existing
# segmentation mask, as we cannot perform inference without any foreground clicks. 


#Lastly, their implementation was presuming that the inference script would be run each iteration rather than being like an app, 
# but they did not provide a mechanism for retaining the lowres mask in some storage, so we will not do that either! 
# We just follow their own logic for retaining the discrete pred mask and resample that into the model domain. Another 
# complication explaining their approach may be that of course the lowres mask is in the model domain, and if you only want 
# to execute inference on a subregion of the full image then it becomes non-trivial to keep the "lowres mask" consistent....
# unlike the 2D slice case where you just execute inference on the slice the prompt is placed on. 




import torch 
import numpy as np
from monai.data import MetaTensor 
import os
import sys 
# basedir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
# sys.path.append(basedir)
app_local_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(app_local_path) 
import copy 

from segment_anything import sam_model_registry
from segment_anything.build_sam3D import sam_model_registry3D
import medim 
# from segment_anything.utils.transforms3D import ResizeLongestSide3D
import torch
import torch.nn.functional as F
import torchio as tio
import re 
import warnings 
import gc 

class InferApp:
    def __init__(self,
                infer_device:torch.device
                ):
        warnings.warn('The SAMMed3D inference app does not yet have a reasonable implementation for outputting the probability map, so the \n' \
        'probability map output will be a dummy tensor of zeros. Please be aware of this when using this app in the validation framework.', UserWarning)
        
        #Hardcoded solution temporary.
        self.infer_device  = infer_device 

        self.app_params = {
            'model_type':'vit_b_ori',
            'checkpoint_name': 'sam_med3d_turbo.pth' #'sam_med3d_turbo_cvpr_coreset.pth' 
            }

        self.load_model()
        self.build_inference_apps()


        self.target_spacing = (1.5, 1.5, 1.5)
        self.patch_size = (128, 128, 128)
        #Some preprocessing, post processing params.
        self.clip_lower_bound = 0.5
        self.clip_upper_bound = 99.5


        WINDOW_LEVEL = 40
        WINDOW_WIDTH = 400 #Values taken from the SegFM challenge organisers, not my own. intended for testing the cvpr ckpt.
        lower_bound = WINDOW_LEVEL - WINDOW_WIDTH / 2
        upper_bound = WINDOW_LEVEL + WINDOW_WIDTH / 2
        #We only use these bounds because this is what the SAMMed3D implementation presumes for the CT images....assuming that
        #we use the cvpr checkpoint. Otherwise we use the default bounds that they used for their default training.
        self.ct_default_bounds = (lower_bound, upper_bound) if self.app_params['checkpoint_name'] == 'sam_med3d_turbo_cvpr_coreset.pth' else (-1000, 1000)  #This is the default bounds for CT images, as per the training script of the SAMMed3D implementation.
        #By default, we do not use the cvpr coreset checkpoint, so we use -1000, 1000 as the default bounds for CT images as taken from their main branch.

        self.permitted_prompts = ('points',)
        self.mask_threshold_sigmoid = 0.5
        self.pre_normalise_bool = True #This is a boolean which determines whether we pre-normalise the image before passing it to the downstream algorithm.
        self.app_params.update({
            'target_spacing': self.target_spacing,
            'model_patch_size': self.patch_size,
            'clip_lower_bound': self.clip_lower_bound,
            'clip_upper_bound': self.clip_upper_bound,
            'ct_default_bounds': self.ct_default_bounds,
            'permitted_prompts': self.permitted_prompts,
            'mask_prob_threshold_sigmoid': self.mask_threshold_sigmoid,
            'pre_normalise_bool': self.pre_normalise_bool,
        })

    def load_model(self):
        # self.sam_model = sam_model_registry3D[self.app_params['model_type']](checkpoint=None).to(
            # self.infer_device
        # )
        
        # self.sam_model = sam_model_registry[self.app_params['model_type
        if self.app_params['checkpoint_name'] is not None:
            
            ckpt_path = os.path.join(app_local_path, 'ckpt', self.app_params['checkpoint_name'])
            
            if self.app_params['checkpoint_name'] == 'sam_med3d_turbo_cvpr_coreset.pth':
                self.model = medim.create_model("SAM-Med3D",
                                                pretrained=True,
                                                checkpoint_path=ckpt_path)
                self.model.to(self.infer_device)
                self.model.eval()
            else:
                #Old versions.
                self.model = sam_model_registry3D[self.app_params['model_type']](checkpoint=None).to(self.infer_device)
                model_dict = torch.load(ckpt_path, map_location=self.infer_device, weights_only=False)
                state_dict = model_dict["model_state_dict"]
                # self.sam_model.load_state_dict(state_dict)
                # self.sam_model.eval()
                self.model.load_state_dict(state_dict)
                self.model.eval()
        else:
            raise Exception 
    
    def build_inference_apps(self):
        #Building the inference app, needs to have an end to end system in place for each "model" type which can be passed by the request: 
        # 
        # IS_autoseg, IS_interactive_init, IS_interactive_edit. (all are intuitive wrt what they represent.) 
        
        self.infer_apps = {
            'IS_autoseg':{'binary_predict':self.binary_inference},
            'IS_interactive_init': {'binary_predict':self.binary_inference},
            'IS_interactive_edit': {'binary_predict':self.binary_inference}
            }
    


    def app_configs(self):

        #STRONGLY Recommended: A method which returns any configuration specific information for printing to the logfile. Expects a dictionary format.
        return self.app_params
    
############################################################################################################################

    def binary_inference(self, request):
        #Mapping the input request to the model domain, this is where we will perform the inference.
        affine, fg_code, perform_inference_bool, roi_im, roi_label, roi_prev_seg, meta_info = self.binary_subject_prep(request=request)

        if perform_inference_bool:
            #perform inference and the post processing. 
            roi_pred = self.sam_model_infer(
                roi_image=roi_im,
                #don't set the prompt generator,just uses the default.
                roi_gt=roi_label,
                prev_low_res_mask=roi_prev_seg #Misnomer on behalf of the authors, as with the roi_gt. Its not the lowres mask.
                #Its just the crop.
                )
            #post process the roi pred for the given iter's interaction into the input image domain scale. 
            pred_ori = self.data_postprocess(
                roi_pred=roi_pred,
                meta_info=meta_info
            )
            #Now we update the segmentation mask in internal storage. This can be used for subsequent calls AND for returning
            #the prediction. 
            self.internal_discrete_output_mask_storage[pred_ori != 0] = fg_code 
        else:
            print('No foreground prompt provided, skipping inference.')
        #If we are not performing inference, then we just return the previous segmentation mask, as there is no foreground prompt to perform inference on.
        
        
        return self.internal_discrete_output_mask_storage.unsqueeze(0), torch.zeros((len(self.configs_labels_dict),) + self.input_dom_im_shape, dtype=torch.float32), affine 
        
    def binary_subject_prep(self, request:dict):
        
        self.dataset_info = request['dataset_info']
        
        if len(self.dataset_info['task_channels']) !=1:
            raise Exception('SAM-Med3D is only supported for single channel images (single modality, modality sequence or pre-fused channels)')
        


        if request['infer_mode'] == 'IS_interactive_edit':
            is_state = request['i_state']
            if all([i is None for i in is_state['interaction_torch_format']['interactions'].values()]) or all([i is None for i in is_state['interaction_torch_format']['interactions_labels'].values()]):
                raise Exception('Cannot be an interactive request without interactive inputs.')
            init = False          
            #The implementation we borrow from does not make any effort to retain their lowres mask in some storage, even 
            #in a file somwhere, so we will not do that either. We always want to be additive, where something is missing, not
            #overwriting the methodology being used.
            assert isinstance(self.resampled_im_subj, tio.Subject) and self.resampled_im_subj
            assert isinstance(self.internal_discrete_output_mask_storage, torch.Tensor) #and self.internal_discrete_output_mask_storage
            # assert isinstance(self.internal_prob_output_mask_storage, torch.Tensor) and self.internal_prob_output_mask_storage
            

        elif request['infer_mode'] == 'IS_interactive_init':
            is_state = request['i_state']
            if all([i is None for i in is_state['interaction_torch_format']['interactions'].values()]) or all([i is None for i in is_state['interaction_torch_format']['interactions_labels'].values()]):
                raise Exception('Cannot be an interactive request without interactive inputs.')
            init = True 
            
            try:
                del self.resampled_im_subj
                del self.internal_discrete_output_mask_storage
                # del self.internal_prob_output_mask_storage
                # gc.collect() #We collect garbage to free up memory, as the internal storage is no longer needed.
                
                #Garbage collection does not seem to be doing anything of worth here, so commenting it out as it slows down the process. 
                torch.cuda.empty_cache() #We clear cache for each new case because image sizes can have variance! 
            except:
                pass #HACK: Not a well engineered solution but want to clear the cached memory while keeping the script "online".
            
            #The implementation we borrow from does not retain their lowres mask in some storage, even 
            #in a file somwhere, so we will not do that either. We always want to be additive, where something is missing, not
            #overwriting the methodology being used.
            

            #Init with empty tensors which will be filled in....
            self.internal_discrete_output_mask_storage = torch.zeros(request['image']['metatensor'].shape[1:], dtype=torch.uint8)
            
            #HACK: We will not be using this in the validation framework just yet. And this is not really used in this algorithm's implementation either, so we will just leave it empty...
            # self.internal_prob_output_mask_storage = torch.zeros((len(request['config_labels_dict']),) + request['image']['metatensor'].shape[1:], dtype=torch.float32)
            #We will just pass through the prob map at the callback return to minimise overhead while we run the simulation. 

            torch.cuda.empty_cache()

        elif request['infer_mode'] == 'IS_autoseg':
            #NOTE: We only put this as a placeholder, autoseg should not be applied to this algorithm as it does not 
            #have the capability, it is way too out of distribution.

            raise NotImplementedError('Autoseg is not supported by this inference app, please use interactive init or edit instead.')

            # key = 'Automatic Init'
            # is_state = request['im'][key]
            # if is_state is not None:
            #     raise Exception('Autoseg should not have any interaction info.')
            # init = True 

            # del self.resampled_im_subj 
            # del self.internal_discrete_output_mask_storage
            # # del self.internal_prob_output_mask_storage
            
            # gc.collect() #We collect garbage to free up memory, as the internal storage is no longer needed.
            # torch.cuda.empty_cache() #We clear cache for each new case because image sizes can have variance! 

            # #The implementation we borrow from does not make any effort to retain their lowres mask in some storage, even 
            # #in a file somwhere, so we will not do that either. We always want to be additive, where something is missing, not
            # #overwriting the methodology being used.

            # #Init with empty tensors which will be filled in....
            # self.internal_discrete_output_mask_storage = torch.zeros(request['image']['metatensor'].shape[1:], dtype=torch.uint8)
            # #We will not really be using this as its not used by the algorithm in any capacity, nor by our validation framework just yet.
            # #So we will just leave it empty and return a dummy tensor in the app callback.
            # # self.internal_prob_output_mask_storage = torch.zeros((len(request['config_labels_dict']),) + request['image']['metatensor'].shape[1:], dtype=torch.float32)
            



        #Mapping image and prompts to the model's coordinate space for inference. 
        
        #NOTE: In order to disentangle the validation framework from inference apps this is always assumed to be handled 
        # within the inference app.

        perform_inference_bool, fg_code, roi_image, roi_label, roi_prev_seg, meta_info = self.binary_prop_to_model(request['image'], is_state, init)        
        
        affine = request['image']['meta_dict']['affine']
        return affine, fg_code, perform_inference_bool, roi_image, roi_label, roi_prev_seg, meta_info  
    
    def binary_prop_to_model(self, im_dict: dict, is_state: dict | None, init: bool):
        #Don't think the init bool will be used much here because there is no way to performance inference on the entire image.
        # so 3D models will always have to use a heuristic to determine the region of interest for inference.

        '''
        Creates the roi image, roi "label" (the region it will use for sampling its own points), and the roi previous seg 
        and meta info for reinserting the segmentation back into image domain. 

        Returns: 
            bool: Whether inference is possible with the provided inputs (i.e. whether there was a foreground prompt). 
            roi_im: the region of interest for the image in model domain. 
            roi_label: the region of interest for the prompt label in model domain.
            roi_prev_seg: the region of interest for the previous segmentation in model domain.
            meta_info: Some meta information about the roi which will be required for reinserting the prediction back into the 
            image domain stored segmentation.
        '''

        #Prompts and images are currently provided in L->R, P->A, I->S, (RAS+ convention). consistent with the torchio 
        #convention also based on nibabel.
    
        #We will structure the order of this code in the same manner as the implementation we borrow from. 
        
        

        #There is one use case for init_bool. The ongoing challenge which the original implementation was designed for had its own
        #image normalisation.


        if init:
            #Here we are following the structure of the read_data_from_npz function. So let us "read". 
            # input_dom_img = im_dict['metatensor']

            #Lets try to minimise the number of variables to prevent memory overhead. 
                
            #The image is assumed to be in 1HWD format, so lets check that it actually has four dimensions.
            if len(im_dict['metatensor'].shape) != 4:
                raise Exception(f'Input image must be in 1HWD format, but got {len(im_dict["metatensor"].shape)} dimensions instead. Please check the input image shape.')
            
            self.input_dom_im_shape = im_dict['metatensor'].shape[1:]
            
            #Now lets extract the image spacing. 
            affine = im_dict['meta_dict']['affine'] 
            if affine is not None:
                if affine.shape[0] != 4: 
                    raise NotImplementedError('We do not yet provide handling for non 3D-annotation domains')
                dim = affine.shape[0] - 1
                _m_key = (slice(-1), slice(-1))
                im_spacing = np.linalg.norm(affine[_m_key] @ np.eye(dim), axis=0)
            else:
                raise Exception('Input image must have an affine array, please check the input image meta_dict.')
        
            if self.pre_normalise_bool:
                #Typically we will pre-clamp/clip on a full image basis as would be expected for both the CVPR SegFM dataset, and the original implementation in the main
                #branch.
                    
                if self.dataset_info['task_channels'][0] == "CT":
            
                    if self.app_params['checkpoint_name'] == 'sam_med3d_turbo_cvpr_coreset.pth':
                        lower_bound, upper_bound = self.ct_default_bounds
                        input_dom_img = np.clip(im_dict['metatensor'].numpy(), lower_bound, upper_bound)
                        input_dom_img = (
                            (input_dom_img - np.min(input_dom_img))
                            / (np.max(input_dom_img) - np.min(input_dom_img) + 1e-6)
                            * 255.0
                        )
                        #Now we will clamp the image's voxel intensity distribution which was not provided in their SegFM script, but in the same vein as the 
                        # # organisers of the SegFM challenge)
                    else:            
                        #We make use of their training script for the normalisation of CT images. They always clamp the image
                        #to -1000 and 1000, so we will do the same.
                        lower_bound, upper_bound = self.ct_default_bounds
                        input_dom_img = np.clip(im_dict['metatensor'].numpy(), lower_bound, upper_bound)
                else:
                    try:
                        if self.app_params['checkpoint_name'] == 'sam_med3d_turbo_cvpr_coreset.pth':
                            lower_bound, upper_bound = np.percentile(
                                im_dict['metatensor'].numpy()[im_dict['metatensor'].numpy() > 0], self.clip_lower_bound
                            ), np.percentile(im_dict['metatensor'].numpy()[im_dict['metatensor'].numpy() > 0], self.clip_upper_bound)
                            input_dom_img = np.clip(im_dict['metatensor'].numpy(), lower_bound, upper_bound)
                            input_dom_img = (
                                (input_dom_img - np.min(input_dom_img))
                                / (np.max(input_dom_img) - np.min(input_dom_img) + 1e-6)
                                * 255.0
                            )
                            input_dom_img[im_dict['metatensor'].numpy() == 0] = 0
                            #Following the SegFM challenge organisers' approach to normalisation for non-CT images. 
                            #Physical interpretation comes into this. For modalities with relative values (i.e. not a fixed physical
                            # interpretation) voxels of <= 0 are typically irrelevant, so they ignored them? 
                        else:
                            #Following an nnU-Net style implementation we will allow the downstream method to handle the 
                            # normalisation with z-score normalisation on the foreground (>0) voxels, as implemented originally.
                            input_dom_img = im_dict['metatensor'].numpy()
                    except:
                        lower_bound, upper_bound = 0, 0
                        #In the case that there were no voxels with intensity > 0, then we just set the lower and upper bounds
                        #to 0 for the cvpr version. For the non-cvpr version, we just pass the image through as-is, the z-score normalisation will
                        #handle the case where the image is all zeros implicitly. 

                input_dom_img = torch.tensor(input_dom_img, dtype=torch.float32)
            else:           
                input_dom_img = im_dict['metatensor'].as_tensor().to(dtype=torch.float32)

            #The original implementation was using a resampling for every single callback, but for some images this could cause
            #slowdown and memory overhead. We will need to be very careful to prevent memory blowup and time blowup.

            #We will resample the image once, and use that for every subsequent callback. For the prev_seg and the point mask however,
            #our hand is forced and we will resample each callback..

            #The resampling ratio is always going to be fixed! 

            self.resampling_ratios = [t/o for o, t in zip(im_spacing, self.target_spacing)]

            #now we will resample the image (resizing technically, because thats what they also did!)
            
            im_subject = tio.Subject(
                image=tio.ScalarImage(tensor=copy.deepcopy(input_dom_img)),
                )
            resampler = tio.Resample(target=self.resampling_ratios)
            self.resampled_im_subj = resampler(im_subject)

            #We will then delete the input_dom_img as we no longer will need it. 
            del input_dom_img 
            del im_subject  
            # gc.collect() #We collect garbage to free up memory, as the input_dom_img is no longer needed.

        

        #Now we propagate the prompt information into the model domain.  

        if bool(is_state):
            
            #First we check whether the interaction mechanism is appropriate.
            p_dict = (is_state['interaction_torch_format']['interactions'], is_state['interaction_torch_format']['interactions_labels'])
    
            #Determine the prompt type from the input prompt dictionaries: Not sure if intersection is optimal for catching exceptions here.
            provided_ptypes = list(set([k for k,v in p_dict[0].items() if v is not None]) & set([k[:-7] for k,v in p_dict[1].items() if v is not None]))
            
            if any([p not in self.permitted_prompts for p in provided_ptypes]):
                raise Exception(f'Non-permitted prompt was supplied, only the following prompts are permitted {self.permitted_prompts}')

            #The SAMMed3D implementation reads in all the prompts in the implementation... but they only perform actions on the 
            # foreground. This is because the logic that they use downstream would break if attempting to use it
            # on the background, e.g.:

            # Their prompt simulation (within the prompt region) would not work. Because they didn't do an actual mapping of the point
            # coordinate to the model domain, as they went through other morphological operations (like resampling etc) and so used a 
            # box mask which they would propagate through to the model domain and then sample a point from that.
            # 
            # When they try to re-insert the changed voxels, they use an assignment according to semantic code. This also wouldn't work. 
            # This would fail if background, as it typically is, has code of 0.  In fact, it cannot  



            # We will follow their logic since we aren't fixing their implementation! 

            #The logic of their code was also written in a way that more than one prompt could not be passed through at the same time.
            #The mechanism that they used was heavily dependent on the simulation strategy of the segFM challenge which presumed that the prompts
            # were always sampled according to the maximal error region.
            # 
            #   - When they read in the clicks, they first append all the foreground, and then all the background for a given semantic class.
            #   - IN THAT ORDER. 
            #   
            #   Then, when they are assessing whether or not to actually perform inference, they look at the semantic code of the final appended
            #   click, and if it is a foreground code, they perform inference. But clearly, this will effectively terminate all refinement once
            #   a background click is appended, as the last click will always be a background click. 

            #   - This is because they do not have a mechanism to determine whether the last click was a foreground or background click,
            

            #This is not going to work for us, as we want to be able to pass both foreground and background clicks at the same time. 

            #So we will just skip over the background clicks, and if there are no foreground clicks we just have to return the existing
            # segmentation mask, as we cannot perform further inference without any foreground clicks. 

            clicks_by_class = is_state['interaction_torch_format']['interactions']['points']
            click_lbs_by_class = is_state['interaction_torch_format']['interactions_labels']['points_labels']

            click_tuple = None #This will be the click tuple that we will use to perform inference. We initialise it to None,
            #in case there are no foreground clicks, in which case we will not perform inference.

            for class_lb, class_code in self.configs_labels_dict.items(): 
                # We write it like this so we don't have to be explicit about the naming convention of the foreground classes.

                if class_lb.title() == 'Background':
                    #We skip the background prompts, as we cannot perform inference on them. 
                    continue 
                else:
                    #We filter to obtain a list of clicks by class code.
                    if len(clicks_by_class) != len(click_lbs_by_class):
                        raise Exception('The number of clicks and click labels must be the same, please check the input prompts.')
                    
                    this_class = [click for i, click in enumerate(clicks_by_class) if click_lbs_by_class[i] == class_code]
                    if this_class == []:
                        continue  #In this case, there was no foreground prompt so we skip. 
                    else:
                        #We will only process the foreground prompts, as these are the ones that we can use to perform inference.
                        
                        #Clicks are in the form of lists of coords for each instance of a prompt. 
                        clicks = this_class 
                        if len(clicks) == 1:
                            #In this case, we have a single point prompt, so we can just use that.
                            #The shape of the click coords is a torch tensor with shape [1, 3]. So we convert to a tuple. 
                            click_coor = tuple(clicks[0][0])
                            click_tuple = (click_coor, [class_code]) #The second element is the label, which is always 1 for foreground clicks.
                            assert class_code == 1 #For binary task it must ONLY be 1 if not background.
                        elif len(clicks) > 1:  
                            raise Exception('This algorithm is not compatible with multiple clicks for a given iteration')
                        else:
                            raise Exception('Should have already flagged the case where len (clicks) == 0')   
        else:
            #Cannot handle empty prompt dictionaries, as this model cannot infer without prompts. 
            raise Exception('Cannot perform inference without prompts')

    
        #Clearly if no click remains (i.e. no foreground), then we early exit. 
        if click_tuple is None:
            return False, None, None, None, None, None  #An unfortunate case of oversegmentation without capability to refine...
        else:
            #Now, they create a "GT" mask centred on the click coordinate, which is what they will use to forward propagate the click
            #information into the model domain, give that they will be performing resampling operations and then sampling from the
            #resampled "click region" to estimate the click position.

            #We will use the default value that the implementation uses for the "gt_roi"! 
            gt_roi = self.create_gt_arr(
                self.input_dom_im_shape, 
                click_tuple[0], 
                click_tuple[1][0])        

            #Now we pass through this and the prev seg mask through for mapping into the model domain.
            # Along with the image which we already resampled.        

            roi_image, roi_label, roi_prev_seg, meta_info = self.data_preprocess(
                img_subj=self.resampled_im_subj,
                cls_gt=gt_roi,
                cls_prev_seg= self.internal_discrete_output_mask_storage,  #This is the previous segmentation mask which we will use to propagate the click information.
                )
            del gt_roi 
            # gc.collect() #We collect garbage to free up memory, as the gt_roi is no longer needed. 

            return True, click_tuple[1][0], roi_image, roi_label, roi_prev_seg, meta_info 
    
    def create_gt_arr(self, shape, point, category_index, square_size=20):
        #This is a function which generates a "ground truth" which outlines a box around a point in the input image domain, so that
        #it can be propagated through the model domain and then sampled from. 
        
        assert isinstance(shape, torch.Size) or isinstance(shape, tuple), 'Shape must be a torch.Size or a tuple'

        # Create an empty tensor with the same shape as the input one.
        gt_array = torch.zeros(shape).to(dtype=torch.uint8) #We don't need float32 here as we are creating a binary mask!
        
        # Extract the coordinates of the point. This algorithm uses RAS+ coordinates, and currently our backend does also. 
        # Therefore we override the reversal implemented in the original implementation which was adapted for an 
        # sitk read image, which would have the ordering in z,y,x (i.e. I-S axis first).
        # 
        x, y, z = point
        
        # Calculate the half size of the square
        half_size = square_size // 2
        
        # Calculate the coordinates of the square around the point
        x_min = max(int(x - half_size), 0)
        x_max = min(int(x + half_size) + 1, shape[0])
        y_min = max(int(y - half_size), 0)
        y_max = min(int(y + half_size) + 1, shape[1])
        z_min = max(int(z - half_size), 0)
        z_max = min(int(z + half_size) + 1, shape[2])
        
        #Their box is actually not even. maybe intentional to help sample from the middle? or maybe just oversight.

        # Set the values within the square to 1
        #gt_array[z_min:z_max, y_min:y_max, x_min:x_max] = category_index
        gt_array[x_min:x_max, y_min:y_max, z_min:z_max] = category_index
        # Note: The order of the dimensions was originally reversed into [Z, Y, X] in the original implementation, 
        # but we use [X, Y, Z] as the coordinates of the prompt are provided in the RAS coordinate system. 

        return gt_array


    def data_preprocess(
            self,
            img_subj, 
            cls_gt, 
            cls_prev_seg
            ):
        
        pred_gt_subject = self.resample_nii(         
            cls_gt, 
            cls_prev_seg, 
            spacing_ratio=self.resampling_ratios)
        
        #merge the img and the pred/gt subjects into one.
        new_subj = tio.Subject(
            image=img_subj.image,
            label=pred_gt_subject.label,
            prev_seg=pred_gt_subject.prev_seg,
            )

        roi_image, roi_label, roi_prev_seg, meta_info = self.read_data_from_subject(new_subj)
    
        meta_info["orig_shape"] = self.input_dom_im_shape
        meta_info["resampled_shape"] = copy.deepcopy(new_subj.spatial_shape),

        del new_subj
        # gc.collect() #We collect garbage to free up memory, as the new_subj is no longer needed.
        #Removed garbage collection as it functionally does not much/anything and just slows down the process.
        return roi_image, roi_label, roi_prev_seg, meta_info


    def resample_nii(
            self,
            gts: torch.Tensor,
            prev_seg: torch.Tensor,
            spacing_ratio: tuple #= (1.5, 1.5, 1.5),
            ):
        """
        Adapted from the SAMMed3D implementation, this function will resample the mask denoting the box around the 
        point and the previous segmentation mask. We already resampled the image as this is fixed and we want to minimise
        runtime and memory overhead.



        #In reality its just performing a resizing operation according to the target spacing and the original spacing.
        #The target_spacing actually provided is a ratio of the input domain spacing, and the actual spacing of the model domain.
        #The tio objects are created without an affine array, so they default to an identity affine, hence why the resampling is just
        #a resizing operation. 


        Parameters:
        - gts: a spatial-shape tensor for the "gt" (i.e. the region around the click).
        - prev_seg: a spatial-shape tensor for the previous seg mask.
        - spacing_ratio: relative ratios between the input and model domain image spacings.... used for resizing the image according to the 
            image spacing itself...
        """
        #Creating the subject object for resampling the prev pred and the mask for propagating the point... 
        subject = tio.Subject(
            label=tio.LabelMap(tensor=gts[None]),
            prev_seg=tio.LabelMap(tensor=prev_seg[None]),
                )
        resampler = tio.Resample(target=spacing_ratio)
        # resampled_subject = resampler(subject)
        return resampler(subject)

    def read_data_from_subject(self,subject):
        '''
        Function for actually extracting the roi according to the mask which is centred on the point! I.e., it will extract a 
        patch where the patch coord is centred around the center of the point box. 
        '''
        crop_transform = tio.CropOrPad(mask_name='label',
                                    target_shape=self.patch_size)#(128, 128, 128))
        padding_params, cropping_params = crop_transform.compute_crop_or_pad(
            subject)
        if (cropping_params is None): cropping_params = (0, 0, 0, 0, 0, 0)
        if (padding_params is None): padding_params = (0, 0, 0, 0, 0, 0)

        #Here they perform a z-score normalisation on the actual sub-region being used for inference! 
        infer_transform = tio.Compose([
            crop_transform,
            tio.ZNormalization(masking_method=lambda x: x > 0),
        ])
        subject_roi = infer_transform(subject)

        # import pdb; pdb.set_trace()
        img3D_roi, gt3D_roi = subject_roi.image.data.clone().detach().unsqueeze(
            1), subject_roi.label.data.clone().detach().unsqueeze(1)
        prev_seg3D_roi = subject_roi.prev_seg.data.clone().detach().unsqueeze(1)
        ori_roi_offset = (
            cropping_params[0],
            cropping_params[0] + 128 - padding_params[0] - padding_params[1],
            cropping_params[2],
            cropping_params[2] + 128 - padding_params[2] - padding_params[3],
            cropping_params[4],
            cropping_params[4] + 128 - padding_params[4] - padding_params[5],
        )

        meta_info = {
            "padding_params": padding_params,
            "cropping_params": cropping_params, 
            "ori_roi": ori_roi_offset,
        }


        del subject 
        del subject_roi #We delete the subject and subject_roi to free up memory, as they are no longer needed. The img3D roi, gt3Droi
        #have been deepcopied and so it will take up more memory! 

        # gc.collect() #We collect garbage to free up memory, as the subject is no longer needed.
        #Removed garbage collection as it functionally does not much/anything and just slows down the process.
        return (
            img3D_roi,
            gt3D_roi,
            prev_seg3D_roi,
            meta_info,
        )
    ##############################################################################################

    #NOTE: This function is going to mess with our own internal seed state.... so we really really need to containerise
    #these applications so that they do not interfere in the future. However, for the initial click it will still be consistent, and subsequent iterations
    #always depend on the previous iteration so they would almost never be consistent across algorithms.
    
    @staticmethod
    def random_sample_next_click(prev_mask, gt_mask):
        """
        Randomly sample one click from ground-truth mask and previous seg mask

        Arguements:
            prev_mask: (torch.Tensor) [H,W,D] previous mask that SAM-Med3D predict
            gt_mask: (torch.Tensor) [H,W,D] ground-truth mask for this image (this is actually the click region)
        """
        #This strategy is lossy given that they only forward propagate the foreground prompts, yet here, they allow for the capability for sampling from the background prompts as well. 

        #In reality, they always pass through an empty mask for the prev mask..... so its only going to ever sample from the foreground.
        
        prev_mask = prev_mask > 0
        true_masks = gt_mask > 0

        if (not true_masks.any()):
            raise ValueError("The resampled box for the point was empty!")

        fn_masks = torch.logical_and(true_masks, torch.logical_not(prev_mask))
        fp_masks = torch.logical_and(torch.logical_not(true_masks), prev_mask)
        
        to_point_mask = torch.logical_or(fn_masks, fp_masks)

        all_points = torch.argwhere(to_point_mask)
        point = all_points[np.random.randint(len(all_points))]

        if fn_masks[point[0], point[1], point[2]]:
            is_positive = True
        else:
            is_positive = False

        sampled_point = point.clone().detach().reshape(1, 1, 3)
        sampled_label = torch.tensor([
            int(is_positive),
        ]).reshape(1, 1)

        return sampled_point, sampled_label

    @torch.no_grad()
    def sam_model_infer(
        self,
        roi_image,
        prompt_generator=random_sample_next_click,
        roi_gt=None,
        prev_low_res_mask=None):
        '''
        Inference for SAM-Med3D, inputs prompt points with its labels (positive/negative for each points)

        # roi_image: (torch.Tensor) cropped image, shape [1,1,128,128,128]
        # prompt_points_and_labels: (Tuple(torch.Tensor, torch.Tensor))
        '''

        device = self.infer_device 
        
        with torch.no_grad(): #might be redundant.
            input_tensor = roi_image.to(device)
            image_embeddings = self.model.image_encoder(input_tensor)

            #memory freed was not adding up when we cleared the image tensor here.... lets do it later so it doesn't interfere with
            #anything by accident.... 

            points_coords, points_labels = torch.zeros(1, 0,
                                                    3).to(device), torch.zeros(
                                                        1, 0).to(device)
            new_points_co, new_points_la = torch.Tensor(
                [[[64, 64, 64]]]).to(device), torch.Tensor([[1]]).to(torch.int64)
            if (roi_gt is not None):
                prev_low_res_mask = prev_low_res_mask if (
                    prev_low_res_mask is not None) else torch.zeros(
                        1, 1, roi_image.shape[2] // 4, roi_image.shape[3] //
                        4, roi_image.shape[4] // 4) #If nonetype then it passes through zeroes.
                prev_low_res_mask = F.interpolate(prev_low_res_mask,
                                                size=(roi_image.shape[2] // 4, roi_image.shape[3] // 4, roi_image.shape[4] // 4),
                                                mode='nearest').to(torch.float32)
                new_points_co, new_points_la = prompt_generator(
                    torch.zeros_like(roi_image)[0, 0], roi_gt[0, 0])
                new_points_co, new_points_la = new_points_co.to(
                    device), new_points_la.to(device)
            points_coords = torch.cat([points_coords, new_points_co], dim=1)
            points_labels = torch.cat([points_labels, new_points_la], dim=1)

            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=[points_coords, points_labels],
                boxes=None,  # we currently not support bbox prompt
                masks=prev_low_res_mask.to(device),
                # masks=None,
            )

            low_res_masks, _ = self.model.mask_decoder(
                image_embeddings=image_embeddings,  # (1, 384, 8, 8, 8)
                image_pe=self.model.prompt_encoder.get_dense_pe(),  # (1, 384, 8, 8, 8)
                sparse_prompt_embeddings=sparse_embeddings,  # (1, 2, 384)
                dense_prompt_embeddings=dense_embeddings,  # (1, 384, 8, 8, 8)
                multimask_output=False,  # They had neglected to include this in their implementation as they used their own package
                #which had pre-set this to False, so we will do that here.
            )

            prev_mask = F.interpolate(low_res_masks,
                                    size=roi_image.shape[-3:],
                                    mode='trilinear',
                                    align_corners=False)

        # convert prob to mask
        medsam_seg_prob = torch.sigmoid(prev_mask)  # (1, 1, 64, 64, 64)
        medsam_seg_prob = medsam_seg_prob.squeeze() #.cpu().squeeze()
        medsam_seg_mask = (medsam_seg_prob > self.mask_threshold_sigmoid).to(dtype=torch.uint8)


        del medsam_seg_prob 
        #taking the input tensor off the device to free up memory, as we will not need it anymore.
        del input_tensor    
        # gc.collect() #We collect garbage to free up memory, as the medsam_seg_prob is no longer needed for now.
        #Garbage collection not doing anything other than slowing things down.
        torch.cuda.empty_cache() #We clear cache for the deleted tensors and for the seg mask.


        return medsam_seg_mask

    def data_postprocess(self, roi_pred, meta_info):

        pred3D_full = torch.zeros(*meta_info["resampled_shape"]).to(device=self.infer_device, dtype=torch.uint8)
        #Creates a full 3D tensor of zeros with the shape of the resampled image. The prediction is re-inserted into this
        # and the resampled back to the original image shape.
        padding_params = meta_info["padding_params"]
                
        #This, I believe, is intended for the case where the patch size was larger than the image patch. 
        unpadded_pred = roi_pred[padding_params[0] : self.patch_size[0] - padding_params[1], #128-padding_params[1],
                                padding_params[2] : self.patch_size[1] - padding_params[3], #128-padding_params[3],
                                padding_params[4] : self.patch_size[2] - padding_params[5] #128-padding_params[5]
                                ].to(device=self.infer_device, dtype=torch.uint8)
        #honestly, not sure, too tired.

        #Now it extracts the offset parameters in the resampled domain and reinserts the prediction here.
        ori_roi = meta_info["ori_roi"]
        pred3D_full[ori_roi[0]:ori_roi[1], ori_roi[2]:ori_roi[3],
                    ori_roi[4]:ori_roi[5]] = copy.deepcopy(unpadded_pred) 
                #We deepcopy so that we can delete the original roi_pred and unpadded pred. otherwise there would still
                # be references.

        pred3D_full_ori = F.interpolate(
            pred3D_full[None][None],
            size=meta_info["orig_shape"],
            mode='nearest').squeeze() #.cpu().squeeze()
        #Squeezes out of all the singletons only leave the spatial dimension. They used nearest neighbour resizing because the tio
        #resampler will not guarantee that the output will have the same shape. This is a very lossy implementation...

        del unpadded_pred 

        # gc.collect() #We collect garbage to free up memory, as the pred3D_full is no longer needed.
        #This garbage collection
        torch.cuda.empty_cache() #We clear cache to free up memory, as the pred3D_full is no longer needed.

        #NOTE: This pred IS JUST FOR THIS ITERATION AND PATCH. We will need to re-insert it into the original prediction mask (i.e. the prev
        #segmentation mask)
        return pred3D_full_ori

    def __call__(self, request: dict):
        
        if len(request['config_labels_dict']) == 2:
            class_type = 'binary'
        elif len(request['config_labels_dict']) > 2:
            class_type = 'multi'
            raise NotImplementedError 
        else:
            raise Exception('Should not have received less than two semantic class labels at minimum')
        
        #We create a duplicate so we can transform the data from metatensor format to the torch tensor format compatible with the inference script.
        modif_request = copy.deepcopy(request) 

        app = self.infer_apps[modif_request['infer_mode']][f'{class_type}_predict']

        #Setting the configs label dictionary for this inference request.
        self.configs_labels_dict = modif_request['config_labels_dict']

        pred, probs_tensor, affine = app(request=modif_request)
        
        pred = pred.to(device='cpu')
        probs_tensor = probs_tensor.to(device='cpu')
        # affine = affine.to(device='cpu')
        torch.cuda.empty_cache()


        #We assert that the output shapes and the affine must match the original metatensor
        assert probs_tensor.shape[1:] == request['image']['metatensor'].shape[1:]
        assert pred.shape[1:] == request['image']['metatensor'].shape[1:] 
        assert torch.all(affine == request['image']['meta_dict']['affine'])
        assert isinstance(probs_tensor, torch.Tensor) 
        assert isinstance(pred, torch.Tensor)
        assert isinstance(affine, torch.Tensor)

        output = {
            'probs':{
                'metatensor':probs_tensor,
                'meta_dict':{'affine': affine}
            },
            'pred':{
                'metatensor':pred,
                'meta_dict':{'affine': affine}
            },
        }

        #Functionally probably wont do anything but putting it here as a placebo. Likely, won't make a diff because there are 
        # still references to these variables throughout.
        del pred 
        del probs_tensor
        del affine
        del modif_request
        # gc.collect() 
        return output 


if __name__ == '__main__':
    infer_app = InferApp(
    torch.device('cuda')
    )

    infer_app.app_configs()

    from monai.transforms import LoadImaged, Orientationd, EnsureChannelFirstd, Compose 
    import nibabel as nib 

    input_dict = {
        #'image':'/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/BraTS2021_Training_Data_Split_True_proportion_0.8_channels_t2_resized_FLIRT_binarised/imagesTs/BraTS2021_00266.nii.gz'
        'image' :os.path.join(app_local_path, 'debug_image/BraTS2021_00266.nii.gz')
        }    
    load_and_transf = Compose([LoadImaged(keys=['image']), EnsureChannelFirstd(keys=['image']), Orientationd(keys=['image'], axcodes='RAS')])
    loaded_im = load_and_transf(input_dict)
    original_affine = loaded_im['image'].meta['original_affine']
    affine = loaded_im['image'].meta['affine'] 
    original_affine = copy.deepcopy(torch.from_numpy(original_affine).to(dtype=torch.float64)) if type(original_affine) is np.ndarray else copy.deepcopy(original_affine.to(dtype=torch.float64))
    affine = copy.deepcopy(torch.from_numpy(affine).to(dtype=torch.float64)) if type(affine) is np.ndarray else copy.deepcopy(affine.to(dtype=torch.float64))
    
    meta = {
        'original_affine': original_affine, 
        'affine': affine
        }
    final_loaded_im = torch.from_numpy(loaded_im['image'].clone().detach().numpy())
    
    request = {
        'image':{
            'metatensor': final_loaded_im,
            'meta_dict':meta
        },
        'infer_mode': 'IS_interactive_init',
        'config_labels_dict':{'background':0, 'tumor':1},
        'dataset_info':{
            'dataset_name':'BraTS2021_t2',
            'dataset_image_channels': {
                "T2w": "0"
            },
            'task_channels': ["T2w"]
        },
        'i_state':
            {
            'interaction_torch_format': {
                'interactions': {
                    'points': [torch.tensor([[40, 103, 43]]), torch.tensor([[61, 62, 39]])],
                    'scribbles': None, 
                    'bboxes': None 
                    },
                'interactions_labels': {
                    'points_labels': [torch.tensor([0]), torch.tensor([1])],
                    'scribbles_labels': None, 
                    'bboxes_labels': None
                    }
                },
            'interaction_dict_format': {
                'points': {'background': [[40, 103, 43]],
                'tumor': [[61,62,39]]
                    },
                'scribbles': None,
                'bboxes': None
                },    
        },
    }
    output = infer_app(request)
    print('halt')