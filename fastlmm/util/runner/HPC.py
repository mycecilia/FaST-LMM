'''
Runs a distributable job on an HPC cluster. Its run method return 'None'

See SamplePi.py for examples.
'''

from fastlmm.util.runner import *
import os
import cPickle as pickle
import subprocess, sys, os.path
import multiprocessing
import fastlmm.util.util as util
import pdb
import logging

class HPC: # implements IRunner
    #!!LATER make it (and Hadoop) work from root directories -- or give a clear error message
    def __init__(self, taskcount, clustername, fileshare, priority="Normal", unit="core", mkl_num_threads=None,
                 remote_python_parent=None,
                update_remote_python_parent=False, min=None, max=None, skipinputcopy=False, logging_handler=logging.StreamHandler(sys.stdout)):
        logger = logging.getLogger()
        if not logger.handlers:
            logger.setLevel(logging.INFO)
        for h in list(logger.handlers):
            logger.removeHandler(h)
        logger.addHandler(logging_handler)
        if logger.level == logging.NOTSET:
            logger.setLevel(logging.INFO)

        self.taskcount = taskcount
        self.clustername = clustername
        self.fileshare = fileshare

        self.priority = priority
        self.unit = unit
        self.min = min
        self.max = max
        self.remote_python_parent = remote_python_parent
        self.update_remote_python_parent = update_remote_python_parent
        self.CheckUnitAndMKLNumThreads(mkl_num_threads, unit)
        self.skipinputcopy=skipinputcopy
      
    def run(self, distributable):
        # Check that the local machine has python path set
        localpythonpath = os.environ.get("PYTHONPATH")#!!should it be able to work without pythonpath being set (e.g. if there was just one file)? Also, is None really the return or is it an exception.
        if localpythonpath == None: raise Exception("Expect local machine to have 'pythonpath' set")

        remotepythoninstall = self.check_remote_pythoninstall()

        remotewd, run_dir_abs, run_dir_rel = self.create_run_dir()
        util.create_directory_if_necessary(os.path.join(remotewd, distributable.tempdirectory), isfile=False) #create temp directory now so that cluster tasks won't try to create it many times at once
        result_remote = os.path.join(run_dir_abs,"result.p")

        self.copy_python_settings(run_dir_abs)

        inputOutputCopier = HPCCopier(remotewd,skipinput=self.skipinputcopy) #Create the object that copies input and output files to where they are needed

        inputOutputCopier.input(distributable) # copy of the input files to where they are needed (i.e. the cluster)

        remotepythonpath = self.FindOrCreateRemotePythonPath(localpythonpath, run_dir_abs)

        batfilename_rel = self.create_bat_file(distributable, remotepythoninstall, remotepythonpath, remotewd, run_dir_abs, run_dir_rel, result_remote)

        self.submit_to_cluster(batfilename_rel, distributable, remotewd, run_dir_abs, run_dir_rel)

        inputOutputCopier.output(distributable) # copy the output file from where they were created (i.e. the cluster) to the local computer

        with open(result_remote, mode='rb') as f:
            result = pickle.load(f)

        logging.info('Done: HPC runner is running a distributable. Returns {0}'.format(result))
        return result



    def CheckUnitAndMKLNumThreads(self, mkl_num_threads, unit):
        if unit.lower() == "core":
            if mkl_num_threads!=None and mkl_num_threads!=1 : raise Exception("When 'unit' is 'core', mkl_num_threads must be unspecified or 1")
            self.mkl_num_threads = 1
        elif unit.lower() == "socket":
            if mkl_num_threads ==None : raise Exception("When 'unit' is 'socket', mkl_num_threads must be specified")
            self.mkl_num_threads = mkl_num_threads
        elif unit.lower() == "node":
            self.mkl_num_threads = mkl_num_threads
        else :
            raise Exception("Expect 'unit' to be 'core', 'socket', or 'node'")

    def copy_python_settings(self, run_dir_abs):
        #localuserprofile = os.environ.get("USERPROFILE")
        user_python_settings=".continuum"
        python_settings=os.path.join(self.fileshare,user_python_settings)
        if os.path.exists(python_settings):
            import shutil
            remote_user_python_settings=os.path.join(run_dir_abs,user_python_settings)
            shutil.copytree(python_settings,remote_user_python_settings)


    def FindOrCreateRemotePythonPath(self, localpythonpath, run_dir_abs):
        if self.remote_python_parent is None:
            remotepythonpath = self.CopySource(localpythonpath, run_dir_abs)
        else:
            util.create_directory_if_necessary(self.remote_python_parent,isfile=False)
            list = []
            for rel in os.listdir(self.remote_python_parent):
                list.append(os.path.join(self.remote_python_parent,rel))
            remotepythonpath = ";".join(list)
            if self.update_remote_python_parent:
                remotepythonpath = self.CopySource(localpythonpath, run_dir_abs)
        
        return remotepythonpath

    def numString(self):
        if self.min == None and self.max == None:
            return " -Num{0} *-*".format(self.unit.capitalize())
        if self.min == None:
            return " -Num{0} {1}".format(self.unit.capitalize(), self.max)
        if self.max == None:
            return " -Num{0} {1}-*".format(self.unit.capitalize(), self.min)
        return " -Num{0} {1}-{2}".format(self.unit.capitalize(), self.min, self.max)

    def submit_to_cluster(self, batfilename_rel, distributable, remotewd, run_dir_abs, run_dir_rel):
        stdout_dir_rel = os.path.join(run_dir_rel,"stdout")
        stdout_dir_abs = os.path.join(run_dir_abs,"stdout")
        util.create_directory_if_necessary(stdout_dir_abs, isfile=False)
        stderr_dir_rel = os.path.join(run_dir_rel,"stderr")
        stderr_dir_abs = os.path.join(run_dir_abs,"stderr")
        util.create_directory_if_necessary(stderr_dir_abs, isfile=False)

        #create the Powershell file
        psfilename_rel = os.path.join(run_dir_rel,"dist.ps1")
        psfilename_abs = os.path.join(run_dir_abs,"dist.ps1")
        util.create_directory_if_necessary(psfilename_abs, isfile=True)
        with open(psfilename_abs, "w") as psfile:
            psfile.write(r"""Add-PsSnapin Microsoft.HPC
        Set-Content Env:CCP_SCHEDULER {0}
        $r = New-HpcJob -Name "{7}" -Priority {8}{12}
        $r.Id
        Add-HpcTask -Name Parametric -JobId $r.Id -Parametric -Start 0 -End {1} -CommandLine "{6} * {5}" -StdOut "{2}\*.txt" -StdErr "{3}\*.txt" -WorkDir {4}
        Add-HpcTask -Name Reduce -JobId $r.Id -Depend Parametric -CommandLine "{6} {5} {5}" -StdOut "{2}\reduce.txt" -StdErr "{3}\reduce.txt" -WorkDir {4}
        Submit-HpcJob -Id $r.Id
        $j = Get-HpcJob -Id $r.Id
        $i = $r.id
        $s = 10

        while(($j.State -ne "Finished") -and ($j.State -ne "Failed") -and ($j.State -ne "Canceled"))
        {10}
            $x = $j.State
            Write-Host "${10}x{11}. Job# ${10}i{11} sleeping for ${10}s{11}"
            Start-Sleep -s $s
            if ($s -ge 60)
            {10}
	        $s = 60
            {11}
            else
            {10}
                $s = $s * 1.1
            {11}
           $j.Refresh()
        {11}

        """                 .format(
                                self.clustername,   #0
                                self.taskcount-1,   #1
                                stdout_dir_rel,     #2
                                stderr_dir_rel,     #3
                                remotewd,           #4
                                self.taskcount,     #5
                                batfilename_rel,    #6
                                self.maxlen(str(distributable),50),      #7
                                self.priority,      #8
                                self.unit,          #9 -- not used anymore,. Instead #12 sets unit
                                "{",                #10
                                "}",                #11
                                self.numString()    #12
                                ))

        import subprocess
        proc = subprocess.Popen(["powershell.exe", "-ExecutionPolicy", "Unrestricted", psfilename_abs], cwd=os.getcwd())
        if not 0 == proc.wait(): raise Exception("Running powershell cluster submit script results in non-zero return code")

    #move to utils?
    @staticmethod
    def maxlen(s,max):
        '''
        Truncate cluster job name if longer than max.
        '''
        if len(s) <= max:
            return s
        else:
            #return s[0:max-1]
            return s[-max:]  #JL: I prefer the end of the name rather than the start

    
    def create_distributablep(self, distributable, run_dir_abs, run_dir_rel):
        distributablep_filename_rel = os.path.join(run_dir_rel, "distributable.p")
        distributablep_filename_abs = os.path.join(run_dir_abs, "distributable.p")
        with open(distributablep_filename_abs, mode='wb') as f:
            pickle.dump(distributable, f, pickle.HIGHEST_PROTOCOL)
        return distributablep_filename_rel

    @staticmethod
    def FindDirectoriesToExclude(localpythonpathdir):
        logging.info("Looking in '{0}' for directories to skip".format(localpythonpathdir))
        xd_string = " /XD $TF"
        for root, dir, files in os.walk(localpythonpathdir):
            for file in files:
             if file.lower() == ".ignoretgzchange":
                 xd_string += " /XD {0}".format(root)
        return xd_string

    def CopySource(self,localpythonpath, run_dir_abs):
        
        if self.update_remote_python_parent:
            remote_python_parent = self.remote_python_parent
        else:
            remote_python_parent = run_dir_abs + os.path.sep + "pythonpath"
        util.create_directory_if_necessary(remote_python_parent, isfile=False)
        remotepythonpath_list = []
        for i, localpythonpathdir in enumerate(localpythonpath.split(';')):
            remotepythonpathdir = os.path.join(remote_python_parent, str(i))
            remotepythonpath_list.append(remotepythonpathdir)
            xd_string = HPC.FindDirectoriesToExclude(localpythonpathdir)
            xcopycommand = 'robocopy /s {0} {1}{2}'.format(localpythonpathdir,remotepythonpathdir,xd_string)
            logging.info(xcopycommand)
            os.system(xcopycommand)

        remotepythonpath = ";".join(remotepythonpath_list)
        return remotepythonpath

    def create_bat_file(self, distributable, remotepythoninstall, remotepythonpath, remotewd, run_dir_abs, run_dir_rel, result_remote):
        path_share_list = [r"",r"Scripts"]
        remotepath_list = []
        for path_share in path_share_list:
            path_share_abs = os.path.join(remotepythoninstall,path_share)
            if not os.path.isdir(path_share_abs): raise Exception("Expect path directory at '{0}'".format(path_share_abs))
            remotepath_list.append(path_share_abs)
        remotepath = ";".join(remotepath_list)

        distributablep_filename_rel = self.create_distributablep(distributable, run_dir_abs, run_dir_rel)

        distributable_py_file = os.path.join(os.path.dirname(__file__),"..","distributable.py")
        if not os.path.exists(distributable_py_file): raise Exception("Expect file at " + distributable_py_file + ", but it doesn't exist.")
        localfilepath, file = os.path.split(distributable_py_file)
        remoteexepath = os.path.join(remotepythonpath.split(';')[0],"fastlmm","util") #!!shouldn't need to assume where the file is in source
        #run_dir_rel + os.path.sep + "pythonpath" + os.path.sep + os.path.splitdrive(localfilepath)[1]

        result_remote2 = result_remote.encode("string-escape")
        command_string = remoteexepath + os.path.sep + file + " " + distributablep_filename_rel + r""" "LocalInParts(%1,{0},mkl_num_threads={1},result_file=""{2}"") " """.format(self.taskcount,self.mkl_num_threads,result_remote2)
        batfilename_rel = os.path.join(run_dir_rel,"dist.bat")
        batfilename_abs = os.path.join(run_dir_abs,"dist.bat")
        util.create_directory_if_necessary(batfilename_abs, isfile=True)
        matplotlibfilename_rel = os.path.join(run_dir_rel,".matplotlib")
        matplotlibfilename_abs = os.path.join(run_dir_abs,".matplotlib")
        util.create_directory_if_necessary(matplotlibfilename_abs, isfile=False)
        util.create_directory_if_necessary(matplotlibfilename_abs + "/tex.cache", isfile=False)
        ipythondir_rel = os.path.join(run_dir_rel,".ipython")
        ipythondir_abs = os.path.join(run_dir_abs,".ipython")
        util.create_directory_if_necessary(ipythondir_abs, isfile=False)
        with open(batfilename_abs, "w") as batfile:
            batfile.write("set path={0};%path%\n".format(remotepath))
            batfile.write("set PYTHONPATH={0}\n".format(remotepythonpath))
            #batfile.write("set R_HOME={0}\n".format(os.path.join(remotepythoninstall,"R-2.15.2")))
            #batfile.write("set R_USER={0}\n".format(remotewd))
            batfile.write("set USERPROFILE={0}\n".format(run_dir_rel))
            batfile.write("set MPLCONFIGDIR={0}\n".format(matplotlibfilename_rel))
            batfile.write("set IPYTHONDIR={0}\n".format(ipythondir_rel))            
            batfile.write("python {0}\n".format(command_string))

        return batfilename_rel

    def check_remote_pythoninstall(self):
        remotepythoninstall = self.fileshare + os.path.sep + "pythonInstallA"
        if not os.path.isdir(remotepythoninstall): raise Exception("Expect Python and related directories at '{0}'".format(remotepythoninstall))

        return remotepythoninstall

    def create_run_dir(self):
        username = os.environ["USERNAME"]
        localwd = os.getcwd()
        #!!make an option to specify the full remote WD. Also what is the "\\\\" case for?
        if localwd.startswith("\\\\"):
            remotewd = self.fileshare + os.path.sep + username +os.path.sep + "\\".join(localwd.split('\\')[4:])
        else:
            remotewd = self.fileshare + os.path.sep + username + os.path.splitdrive(localwd)[1]  #using '+' because 'os.path.join' isn't work with shares
        import datetime
        now = datetime.datetime.now()
        run_dir_rel = os.path.join("runs",util.datestamp(appendrandom=True))
        run_dir_abs = os.path.join(remotewd,run_dir_rel)
        util.create_directory_if_necessary(run_dir_abs,isfile=False)
        return remotewd, run_dir_abs, run_dir_rel


class HPCCopier(object): #Implements ICopier

    def __init__(self, remotewd, skipinput=False):
        self.remotewd = remotewd
        self.skipinput=skipinput

    def input(self,item):
        if self.skipinput:
            return
        if isinstance(item, str):
            itemnorm = os.path.normpath(item)
            remote_file_name = os.path.join(self.remotewd,itemnorm)
            remote_dir_name,ignore = os.path.split(remote_file_name)
            util.create_directory_if_necessary(remote_file_name)
            xcopycommand = "xcopy /d /e /s /c /h /y {0} {1}".format(itemnorm, remote_dir_name)
            logging.info(xcopycommand)
            rc = os.system(xcopycommand)
            print "rc=" +str(rc)
            if rc!=0: raise Exception("xcopy cmd failed with return value={0}, from cmd {1}".format(rc,xcopycommand))
        elif hasattr(item,"copyinputs"):
            item.copyinputs(self)
        # else -- do nothing

    def output(self,item):
        if isinstance(item, str):
            itemnorm = os.path.normpath(item)
            util.create_directory_if_necessary(itemnorm)
            remote_file_name = os.path.join(self.remotewd,itemnorm)
            local_dir_name,ignore = os.path.split(itemnorm)
            #xcopycommand = "xcopy /d /e /s /c /h /y {0} {1}".format(remote_file_name, local_dir_name) # we copy to the local dir instead of the local file so that xcopy won't ask 'file or dir?'
            xcopycommand = "xcopy /d /c /y {0} {1}".format(remote_file_name, local_dir_name) # we copy to the local 
            logging.info(xcopycommand)
            rc = os.system(xcopycommand)
            if rc!=0: logging.info("xcopy cmd failed with return value={0}, from cmd {1}".format(rc,xcopycommand))
        elif hasattr(item,"copyoutputs"):
            item.copyoutputs(self)
        # else -- do nothing
