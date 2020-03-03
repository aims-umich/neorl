import numpy as np
import os

#--------------------------------------------------------------------------------------------------
# This class intialize the coupling step by writing the first input file for the coupling scheme
# It contains all methods for all codes that take initial conditions from the master input and write the 
# first input file
#--------------------------------------------------------------------------------------------------     

#from PARSER import PARSER
 
class FirstStep:
    def __init__(self, pars1):
        self.pars1=pars1
            
    #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # TRACE
    #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    def WriteTrace(self,TotPower,trace_inname):
        #reading the template file into a string bracket_count
        with open(self.pars1.t["TraceInputName"],"r") as template:
            template_text = template.read()
        power_count = template_text.count("AxialPower")
        
        #checking to make sure there is the proper number of nodes in the input file compared to the template
        if not self.pars1.t["NodeID"].size == power_count:
            raise Exception("Brackets in TRACE input should not be more than the number of nodes in trace NodeID")
        
        #Find index of each Axial Power in the template
        for i in range(len(self.pars1.t["NodeID"])):
            template_text = template_text.replace("{AxialPower" + str(self.pars1.t["NodeID"][i]) + "}", str(self.pars1.t["AxialPower"][i]))
        
        # for transient mode, the power changes
        template_text=template_text.replace("{Power}", str(np.round(TotPower,3))+'E6')
            
        with open(trace_inname, "w") as outfile:
            outfile.write(template_text)
            
    #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # RELAP5
    #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    
    def WriteRelap5(self,relap5_inname):
        #reading the template file into a string bracket_count
        with open(self.pars1.r5["Relap5InputName"],"r") as template:
            template_text = template.read()
        power_count = template_text.count("AxialPower")
        
        #checking to make sure there is the proper number of nodes in the input file compared to the template
        if not self.pars1.r5["NodeID"].size == power_count:
            raise Exception("Brackets in RELAP5 input should not be more than the number of nodes in relap5 NodeID")
            
        #Simple replace axial power in template
        #Find index of each Axial Power in the template
        for i in range(len(self.pars1.r5["NodeID"])):
            template_text = template_text.replace("{AxialPower" + str(self.pars1.r5["NodeID"][i]) + "}", str(self.pars1.r5["AxialPower"][i]))
            
        with open(relap5_inname, "w") as outfile:
            outfile.write(template_text)
    #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # BISON
    #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
       
    def WriteBison(self, bison_inname,csvpower_inname, StateFile, Time):
        #determine the communication mode with BISON
        if (os.path.exists(self.pars1.bs["AuxPowerFile"])):
            #reading the axial power template file from auxiliary csv or txt file
            with open(self.pars1.bs["AuxPowerFile"],"r") as template:
                template_power = template.read()
        else:
            raise Exception ('the power csv template file {parm} for bison does not exist'.format(parm = self.pars1.bs["AuxPowerFile"]))

        with open(self.pars1.bs["BisonInputName"],"r") as template:
            template_bison = template.read()

            
        power_count = template_power.count("AxialPower")
        
        aux_csv_count = template_bison.count("AuxPowerFile")  # in bison template
        prevstate_count = template_bison.count("PrevState") # in bison template
        nextstate_count = template_bison.count("NextState") # in bison template
        starttime_count = template_bison.count("StartTime") # in bison template
        endtime_count = template_bison.count("EndTime") # in bison template
        #checking to make sure there is the proper number of nodes in the input file compared to the template
        if not 2*self.pars1.bs["NodeID"].size == power_count:
            raise Exception("Brackets in the BISON aux power file should not be more than twice the number of nodes in bison NodeID")
        if aux_csv_count == 0:
            raise Exception("Brackets in BISON input for the csv power file don't exist, at least one bracket {AuxPowerFile} should exist")
        
        #Find index of each Axial Power in the template
        for i in range(len(self.pars1.bs["NodeID"])):
            template_power = template_power.replace("{AxialPower" + str(self.pars1.bs["NodeID"][i]) + "}", str(self.pars1.bs["AxialPower"][i]))
        # Replace Time variables in the csv 
        template_power = template_power.replace("{StartTime}",str(Time[0]))
        template_power = template_power.replace("{EndTime}",str(Time[1]))
        
        for i in range(aux_csv_count):
            template_bison = template_bison.replace("{AuxPowerFile}",csvpower_inname)
            
        if (prevstate_count > 0):
            template_bison = template_bison.replace("{PrevState}",StateFile[1])
            
        if (nextstate_count > 0):
            template_bison = template_bison.replace("{NextState}",StateFile[0])
        
        if (starttime_count > 0):
            template_bison = template_bison.replace("{StartTime}",str(Time[0]))
        
        if (endtime_count > 0):
            template_bison = template_bison.replace("{EndTime}",str(Time[1]))
            
        with open(csvpower_inname, "w") as outfile:
            outfile.write(template_power)
            
        with open(bison_inname, "w") as outfile:
            outfile.write(template_bison)
    
    #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # SCALE (unused)
    #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    
    #-----------------------------------------------------------------------------------------------
    # ! the function below is not used as starting with the neutronics code is not a good practice 
    # ---------------------------------------------------------------------------------------------
    def WriteScale(self, scale_new_name):
        #reading the template file
        with open(self.pars1.s["ScaleInputName"], "r") as template:
            template_text = template.read()
        
        #Find index of each Axial Power in the template
        for i in range(len(self.pars1.s["NodeID"])):
            template_text = template_text.replace("{CoolDens" + str(self.pars1.s["NodeID"][i]) + "}", str(self.pars1.s["CoolDens"][i])) 
            template_text = template_text.replace("{CoolTemp" + str(self.pars1.s["NodeID"][i]) + "}", str(self.pars1.s["CoolTemp"][i]))

        with open(scale_new_name, "w") as outfile:
            outfile.write(template_text)