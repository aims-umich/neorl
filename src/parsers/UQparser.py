import numpy as np
from scipy.stats import truncnorm, norm
import os
import shutil
from distutils.dir_util import copy_tree

class UQparser:
    def __init__(self, NT1):
        def get_truncated_normal(mean, sd, low, upp):
            return truncnorm(
                    (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
            
        def inplace_change(filename, old_string, new_string):
            # Safely read the input filename using 'with'
            with open(filename) as f:
                s = f.read()
                if old_string not in s:
                    print('"{old_string}" not found in {filename}.'.format(**locals()))
                    return
        
            # Safely write the changed content, if found in the file
            with open(filename, 'w') as f:
                s = s.replace(old_string, new_string)
                f.write(s)
                
        def line_prepender(filename, line):
            with open(filename, 'r+') as f:
                content = f.read()
                f.seek(0, 0)
                f.write(line.rstrip('\r\n') + '\n' + content)
            
        working_directory = "./"
        
        perturbed_xs_string="""=shell
  ln -sf ${DATA}/scale.rev04.xn56v7.1 ft89f001
  ln -sf ${DATA}/perturb/56n.v7.1/Sample{{id}} ft87f001
end 

=clarolplus
  in=89
  out=88
  var=87
  isvar=10
  bond=yes
  sam={{id}}
end
=shell
  mv ft88f001 v7-56
  unlink ft87f001
  unlink ft89f001
  mv ft10f001 crawdadPerturbMGLib
end
"""

        perturbed_decay_string = """=shell
  ln -sf ${DATA}/perturb/end7dec_{{id}}  end7dec
end
"""

        perturbed_yield_string="""=YieldSampler
  pertnum={{id}}
end
=shell
  ln -sf  yields  yield.data 
end
"""
        
        template_directory_list = [] 
        caseids = []
        
        #getting number of samples
        n = [int(line.split("=")[1].strip()) for line in NT1.uq_block if line.split("=")[0].strip() == "n_samples" ][0]
        p_xs=[str(line.split("=")[1].strip()) for line in NT1.uq_block if line.split("=")[0].strip() == "perturb_xs" ][0]
        p_yield=[str(line.split("=")[1].strip()) for line in NT1.uq_block if line.split("=")[0].strip() == "perturb_yield" ][0]
        p_decay=[str(line.split("=")[1].strip()) for line in NT1.uq_block if line.split("=")[0].strip() == "perturb_decay" ][0]
        
        #Creating directory structure to put in sampled template files
        try:
            os.mkdir(working_directory + NT1.CaseName + "_tsamples")
        except:
            pass
        
        for i in range(0,n+1):
            caseid = str(i).zfill(4)
            caseids.append(caseid)
            try:
                os.mkdir(working_directory + NT1.CaseName + "_tsamples" + "/" + "template_sample" + caseid)
            except:
                pass
            
            template_directory_list.append(working_directory + NT1.CaseName + "_tsamples" + "/" + "template_sample" + caseid)
            
        #removing UQ block from master input ant storing it as list of strings     
        with open(NT1.input_file_path) as master:
            filetext = master.readlines()
            
        start = [i for i, s in enumerate(filetext) if 'READ UQ' in s][0]
        end = [i for i, s in enumerate(filetext) if 'END UQ' in s][0]
        for i in range(start, end+1):
            filetext[i] = "%" + filetext[i]

        for directory in template_directory_list:
#            shutil.copy("fuse.py", directory)    
#            os.system("cp -r src " + directory)       
            with open(directory + "/" + NT1.input_file_path,"w") as out:
                out.write("".join(filetext))
                
            
            
        #Copying braketed template into each directory in template_directory_list
        scale_template_list = []
        trace_template_list = []
        relap5_template_list = []
        if NT1.ScaleFlag:
            for i in range(len(template_directory_list)):
                casedir = template_directory_list[i] + "/" + NT1.s["ScaleInputName"]
                shutil.copyfile(NT1.s["ScaleInputName"], casedir)
                plib_id = template_directory_list[i][-4:]
                # Check perturb XS
                if(p_yield):
                    line_prepender(template_directory_list[i] + "/" + NT1.s["ScaleInputName"], perturbed_yield_string.replace("{{id}}",str(i+1)))
                if (p_decay=='yes'):
                    line_prepender(template_directory_list[i] + "/" + NT1.s["ScaleInputName"], perturbed_decay_string.replace("{{id}}",plib_id))
                if (p_xs=='yes'):
                    line_prepender(template_directory_list[i] + "/" + NT1.s["ScaleInputName"], perturbed_xs_string.replace("{{id}}",str(i+1)))
                scale_template_list.append(casedir)
        if NT1.TraceFlag:
            for i in range(len(template_directory_list)):
                casedir = template_directory_list[i] + "/" + NT1.t["TraceInputName"]
                shutil.copyfile(NT1.t["TraceInputName"], casedir)
                trace_template_list.append(casedir)
        if NT1.Relap5Flag:
            for i in range(len(template_directory_list)):
                casedir = template_directory_list[i] + "/" + NT1.r5["Relap5InputName"]
                shutil.copyfile(NT1.r5["Relap5InputName"], casedir)
                relap5_template_list.append(casedir)
            
        
        for key, value in NT1.uq.items():
            #aliasing code-specific template to general variable
            if value[0].strip() == "scale":
                tlist = scale_template_list
            if value[0].strip() == "trace":
                tlist = trace_template_list
            if value[0].strip() == "relap5":
                tlist = relap5_template_list

            if value[1].strip() == "norm":
                dist = norm(loc = float(value[2]), scale = float(value[3]))
            if value[1].strip() == "truncnorm":
                dist = get_truncated_normal(mean = float(value[2]), sd = float(value[3]), low = float(value[4]), upp = float(value[5]))
                
            for template in tlist:
                if (dist.rvs()>=10**3):
                    inplace_change(template, "[[" + key + "]]", "%.3e"%(dist.rvs()))
                else:
                    inplace_change(template, "[[" + key + "]]", "%.4f"%(dist.rvs()))
