import os
import numpy as np
import sys
import scipy.io
from src.data.parser import ParserPCG

class BuildTimeFeatures(ParserPCG):

    def __init__(self, basepath, endpath):
        super.__init__(basepath, endpath)

        # Number of features generate by get_time_features function.
        # this needs to be seted in order to create the final matrix (X) conteining all features extracted from all
        # wav files
        self.nfeatures = 20

    def get_time_features(self, PCG, assigned_states):
        # We just assume that the assigned_states cover at least 2 whole heart beat cycle
        indx = np.where(np.abs(np.diff(assigned_states)) > 0)[0]  # find the locations with changed states

        if assigned_states[0] > 0:  # for some recordings, there are state zeros at the beginning of assigned_states
            if assigned_states[0] == 4:
                K = 0
            elif assigned_states[0] == 3:
                K = 1
            elif assigned_states[0] == 2:
                K = 2
            elif assigned_states[0] == 1:
                K = 3
                pass
        else:
            if assigned_states[indx[0] + 1] == 4:
                K = 0
            elif assigned_states[indx[0] + 1] == 3:
                K = 1
            elif assigned_states[indx[0] + 1] == 2:
                K = 2
            elif assigned_states[indx[0] + 1] == 1:
                K = 3
                pass
            K = K + 1

        indx = indx[K:]  # From de K state (always 4 state ) to the end

        # delete items from indx to get 4 state complete
        rem = np.mod(len(indx), 4)

        indx = indx[:len(indx) - rem]

        # A is N*4 matrix, the 4 columns save the beginnings of S1, systole, S2 and diastole in the same heart cycle respectively
        A = np.reshape(indx, (int(len(indx) / 4), 4))

        # Feature calculation
        m_RR = np.round(np.mean(np.diff(A[:, 0])))  # mean value of RR intervals
        sd_RR = np.round(np.std(np.diff(A[:, 0])))  # standard deviation (SD) value of RR intervals
        mean_IntS1 = np.round(np.mean(A[:, 1] - A[:, 0]))  # np.mean value of S1 intervals
        sd_IntS1 = np.round(np.std(A[:, 1] - A[:, 0]))  # SD value of S1 intervals
        mean_IntS2 = np.round(np.mean(A[:, 3] - A[:, 2]))  # np.mean value of S2 intervals
        sd_IntS2 = np.round(np.std(A[:, 3] - A[:, 2]))  # SD value of S2 intervals
        mean_IntSys = np.round(np.mean(A[:, 2] - A[:, 1]))  # np.mean value of systole intervals
        sd_IntSys = np.round(np.std(A[:, 2] - A[:, 1]))  # SD value of systole intervals
        mean_IntDia = np.round(np.mean(A[1:, 0] - A[0:-1, 3]))  # np.mean value of diastole intervals
        sd_IntDia = np.round(np.std(A[1:, 0] - A[0:-1, 3]))  # SD value of diastole intervals

        R_SysRR = []
        R_DiaRR = []
        R_SysDia = []
        P_S1 = []
        P_Sys = []
        P_S2 = []
        P_Dia = []
        P_SysS1 = []
        P_DiaS2 = []

        for i in range(0, A.shape[0] - 1, 1):
            R_SysRR.append((A[i, 2] - A[i, 1]) / (A[i + 1, 0] - A[i, 0]) * 100)
            R_DiaRR.append((A[i + 1, 0] - A[i, 3]) / (A[i + 1, 0] - A[i, 0]) * 100)
            R_SysDia.append(R_SysRR[i] / R_DiaRR[i] * 100)
            P_S1.append(np.sum(np.abs(PCG[A[i, 0]:A[i, 1]])) / (A[i, 1] - A[i, 0]))
            P_Sys.append(np.sum(np.abs(PCG[A[i, 1]:A[i, 2]])) / (A[i, 2] - A[i, 1]))
            P_S2.append(np.sum(np.abs(PCG[A[i, 2]:A[i, 3]])) / (A[i, 3] - A[i, 2]))
            P_Dia.append(np.sum(abs(PCG[A[i, 3]:A[i + 1, 0]])) / (A[i + 1, 0] - A[i, 3]))

            if P_S1[i] > 0:
                P_SysS1.append(P_Sys[i] / P_S1[i] * 100)
            else:
                P_SysS1.append(0)

            if P_S2[i] > 0:
                P_DiaS2.append(P_Dia[i] / P_S2[i] * 100)
            else:
                P_DiaS2.append(0)

        R_SysRR = np.asarray(R_SysRR)
        R_DiaRR = np.asarray(R_DiaRR)
        R_SysDia = np.asarray(R_SysDia)
        P_S1 = np.asarray(P_S1)
        P_Sys = np.asarray(P_Sys)
        P_S2 = np.asarray(P_S2)
        P_Dia = np.asarray(P_Dia)
        P_SysS1 = np.asarray(P_SysS1)
        P_DiaS2 = np.asarray(P_DiaS2)

        m_Ratio_SysRR = np.mean(R_SysRR)  # mean value of the interval ratios between systole and RR in each heart beat
        sd_Ratio_SysRR = np.std(R_SysRR)  # SD value of the interval ratios between systole and RR in each heart beat
        m_Ratio_DiaRR = np.mean(R_DiaRR)  # mean value of the interval ratios between diastole and RR in each heart beat
        sd_Ratio_DiaRR = np.std(R_DiaRR)  # SD value of the interval ratios between diastole and RR in each heart beat
        m_Ratio_SysDia = np.mean(
            R_SysDia)  # mean value of the interval ratios between systole and diastole in each heart beat
        sd_Ratio_SysDia = np.std(
            R_SysDia)  # SD value of the interval ratios between systole and diastole in each heart beat

        indx_sys = np.where(P_SysS1 > 0) and np.where(P_SysS1 < 100)[0]  # avoid the flat line signal

        if indx_sys.__len__() > 1:
            m_Amp_SysS1 = np.mean(P_SysS1[
                                      indx_sys])  # mean value of the mean absolute amplitude ratios between systole period and S1 period in each heart beat
            sd_Amp_SysS1 = np.std(P_SysS1[
                                      indx_sys])  # SD value of the mean absolute amplitude ratios between systole period and S1 period in each heart beat
        else:
            m_Amp_SysS1 = 0
            sd_Amp_SysS1 = 0

        indx_dia = np.where(P_DiaS2 > 0) and np.where(P_DiaS2 < 100)[0]
        if indx_dia.__len__() > 1:
            m_Amp_DiaS2 = np.mean(P_DiaS2[
                                      indx_dia])  # mean value of the mean absolute amplitude ratios between diastole period and S2 period in each heart beat
            sd_Amp_DiaS2 = np.std(P_DiaS2[
                                      indx_dia])  # SD value of the mean absolute amplitude ratios between diastole period and S2 period in each heart beat
        else:
            m_Amp_DiaS2 = 0
            sd_Amp_DiaS2 = 0

        return [m_RR, sd_RR, mean_IntS1, sd_IntS1, mean_IntS2, sd_IntS2, mean_IntSys, sd_IntSys, mean_IntDia, sd_IntDia,
                m_Ratio_SysRR, sd_Ratio_SysRR, m_Ratio_DiaRR, sd_Ratio_DiaRR, m_Ratio_SysDia, sd_Ratio_SysDia,
                m_Amp_SysS1, sd_Amp_SysS1, m_Amp_DiaS2, sd_Amp_DiaS2]

    def load(self):
        """
        Loads physio 2016 challenge dataset from self.basepath by crawling the path.
        For each discovered mat file:

        * Attempt to parse the header file for class label
        * Attempt to load the mat file

        Returns
        -------
        None
        """

        # First pass to calculate number of samples
        # ensure each wav file has an associated and parsable
        # Header file
        mat_file_names = []
        class_labels = []
        for root, dirs, files in os.walk(self.basepath):
            # Ignore validation for now!
            if "validation" in root:
                continue
            for file in files:
                if file.endswith('.mat'):
                    try:
                        base_file_name = file.rstrip(".mat")
                        label_file_name = os.path.join(root, base_file_name + ".hea")

                        class_label = self.__parse_class_label(label_file_name)
                        class_labels.append(self.class_name_to_id[class_label])
                        mat_file_names.append(os.path.join(root, file))

                        self.n_samples += 1
                    except InvalidHeaderFileException as e:
                        print(e)

        # Inicialize X zeros array
        X = np.zeros([self.n_samples, self.nfeatures])

        for idx, matfname in enumerate(mat_file_names):

            # read the mat file
            matfile = scipy.io.loadmat(matfname)

            PCG = matfile['out'][:, 0]
            assigned_states = matfile['out'][:, 1]

            # gets features from each pcg file
            features = self.get_time_features(PCG, assigned_states)

            # saving on final X matrix
            X[idx, :] = features

            idx += 1

        self.X = X

        class_labels = np.array(class_labels)

        # Map from dense to one hot
        self.y = np.eye(self.nclasses)[class_labels]

    def save(self, save_path):
        """
        Persist the PCG features and class to disk

        Parameters
        ----------
        save_path: str
            Location on disk to store the parsed PCG's features metadata

        Returns
        -------
        None

        """
        np.save(os.path.join(save_path, "X_TF.npy"), self.X)
        np.save(os.path.join(save_path, "y.npy"), self.y)





# matfile = scipy.io.loadmat('/Users/guillelissa/Projects/DeepHeartSound/data/segmented/a0222.mat')
#
# PCG = matfile['out'][:, 0]
# assigned_states = matfile['out'][:, 1]
#
# features = get_time_features(PCG, assigned_states)
# print(features)