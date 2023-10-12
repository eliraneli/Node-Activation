import numpy as np
import torch

DEBUG = False
TRAINING = True
SUM_PRODUCT = False
MIN_SUM = not SUM_PRODUCT
ALL_ZEROS_CODEWORD_TRAINING = False
ALL_ZEROS_CODEWORD_TESTING = False
NO_SIGMA_SCALING_TRAIN = False
NO_SIGMA_SCALING_TEST = False
np.set_printoptions(precision=3)


class Decoder(torch.nn):
    def __init__(self, batch_size, num_iterations, code, device, decoder_type="RNOMS", random_seed=0,
                 learning_rate=0.001, relaxed=False):
        self.decoder_type = decoder_type
        self.random_seed = random_seed
        self.learning_rate = learning_rate
        self.relaxed = relaxed
        self.batch_size = batch_size
        self.num_iterations = num_iterations
        self.code = code
        self.device = device
        self.init_decoder(self.num_iterations)

    def init_decoder(self, num_iterations):
        if SUM_PRODUCT:
            if self.decoder_type == "FNSPA":
                # maybe this torch.fmod(torch.randn([num_w1, num_w2]),1) - keep sign or maybe rand
                self.W_cv = torch.normal(mean=0, std=1, size=(num_iterations, self.code.num_edges))

            if self.decoder_type == "RNSPA":
                self.W_cv = torch.normal(mean=0, std=1, size=self.code.num_edges)

        if MIN_SUM:
            if self.decoder_type == "FNNMS":
                self.W_cv = torch.normal(mean=0, std=1, size=(num_iterations, self.code.num_edges))

            if self.decoder_type == "FNOMS":
                self.B_cv = torch.normal(mean=0, std=1, size=(num_iterations, self.code.num_edges))
                self.D_cv = torch.normal(mean=0, std=1, size=(num_iterations, self.code.num_edges))

            if self.decoder_type == "RNNMS":
                self.W_cv = torch.nn.Softplus(torch.normal(mean=0, std=1, size=self.code.num_edges))

            if self.decoder_type == "RNOMS":
                self.B_cv = torch.normal(mean=0, std=1, size=self.code.num_edges)
                self.D_cv = torch.normal(mean=0, std=1, size=self.code.num_edges)

        if self.relaxed:
            self.R = 0.5

    # combine messages to get posterior LLRs
    def marginalize(self, soft_input, cv):

        weighted_soft_input = soft_input

        soft_output = torch.tensor([]).to(self.device)
        for i in range(0, self.code.n):
            edges = []
            for e in range(0, self.code.var_degrees[i]):
                edges.append(self.code.d[i][e])

            temp = cv[self.code.edges_m[i], :]
            temp = torch.sum(temp, 0).to(self.device)
            soft_output = torch.cat((soft_output, temp), 0)

        soft_output = torch.reshape(soft_output, soft_input.size())
        soft_output = weighted_soft_input + soft_output

        return soft_output

    # compute messages from variable nodes to check nodes
    def compute_vc(self, cv, soft_input):

        weighted_soft_input = soft_input
        reordered_soft_input = weighted_soft_input[self.code.edges, :]

        vc = []
        # for each variable node v, find the list of extrinsic edges
        # fetch the LLR values from cv using these indices
        # find the sum and store in temp, finally store these in vc
        count_vc = 0
        for i in range(0, self.code.n):
            for j in range(0, self.code.var_degrees[i]):
                # if the list of extrinsic edges is not empty, add them up
                if self.code.extrinsic_edges_vc[count_vc]:
                    temp = cv[self.code.extrinsic_edges_vc[count_vc], :]
                    temp = torch.sum(temp, 0)
                else:
                    temp = torch.zeros([self.batch_size])
                vc.append(temp.to(self.device))
                count_vc = count_vc + 1
        vc = torch.stack(vc)
        new_order_vc = np.zeros(self.code.num_edges).astype(int)
        new_order_vc[self.code.edge_order_vc] = np.array(range(0, self.code.num_edges)).astype(int)
        vc = vc[new_order_vc, :]

        return vc.to(self.device) + reordered_soft_input

    # compute messages from check nodes to variable nodes
    def compute_cv(self, vc, iteration, isRelu):

        cv_list = torch.tensor([]).to(self.device)
        prod_list = torch.tensor([]).to(self.device)
        min_list = torch.tensor([]).to(self.device)



        if SUM_PRODUCT:
            vc = torch.clip(vc, -10, 10)
            tanh_vc = torch.tanh(vc / 2.0)


        count_cv = 0
        for i in range(0, self.code.m):  # for each check node c
            for j in range(0, self.code.chk_degrees[i]):
                if SUM_PRODUCT:
                    temp = tanh_vc[self.code.extrinsic_edges_cv[count_cv], :]
                    temp = torch.prod(temp, 0)
                    temp = torch.log((1 + temp) / (1 - temp))
                    cv_list = torch.cat((cv_list, temp.float()), 0)

                if MIN_SUM:
                    if self.code.extrinsic_edges_cv[count_cv]:
                        temp = vc[self.code.extrinsic_edges_cv[count_cv], :]
                    else:
                        temp = torch.zeros([1, self.batch_size]).to(self.device)

                    prod_chk_temp = torch.prod(torch.sign(temp), 0)
                    (sign_chk_temp, min_ind) = torch.min(torch.abs(temp), 0)
                    prod_list = torch.cat((prod_list, prod_chk_temp.float()), 0)
                    min_list = torch.cat((min_list, sign_chk_temp.float()), 0)
                count_cv += 1

        if SUM_PRODUCT:
            cv = torch.reshape(cv_list, vc.size())
        if MIN_SUM:
            prods = torch.reshape(prod_list, vc.size())  # stack across batch size
            mins = torch.reshape(min_list, vc.size())


            soft_plus_operator = torch.nn.Softplus()

            if self.decoder_type == "RNOMS":

                offsets = soft_plus_operator(self.B_cv)
                mins *= torch.tile(torch.reshape(offsets, [-1, 1]), [1, self.batch_size]).to(self.device)


            elif self.decoder_type == "FNOMS":

                offsets = torch.tile(torch.reshape(soft_plus_operator(self.B_cv[iteration]), [-1, 1]), [1, self.batch_size]).to(self.device)
                if isRelu:
                    mins = torch.nn.functional.relu(mins - offsets)
                else:
                    mins *= offsets
            cv = prods * mins

        new_order_cv = np.zeros(self.code.num_edges).astype(np.int)
        new_order_cv[self.code.edge_order_cv] = np.array(range(0, self.code.num_edges)).astype(int)
        cv = cv[new_order_cv, :]

        if self.decoder_type == "RNSPA" or self.decoder_type == "RNNMS":
            cv *= torch.tile(torch.reshape(self.W_cv, [-1, 1]), [1, self.batch_size]).to(self.device)
        elif self.decoder_type == "FNSPA" or self.decoder_type == "FNNMS":
            cv *= torch.tile(torch.reshape(self.W_cv[iteration], [-1, 1]), [1, self.batch_size]).to(self.device)
        return cv

    def belief_propagation_iteration(self, soft_input, iteration, cv, m_t, isRelu):
        # compute vc
        vc = self.compute_vc(cv, soft_input)
        # filter vc
        if self.relaxed:
            m_t = self.R * m_t + (1 - self.R) * vc
            vc_prime = m_t
        else:
            vc_prime = vc
        # compute vc
        cv = self.compute_cv(vc_prime, iteration, isRelu)
        soft_output = self.marginalize(soft_input, cv)
        iteration += 1

        return soft_input, soft_output, iteration, cv, m_t, vc



    # compute the "soft syndrome"
    def syndrome(self, soft_output, code):
        soft_syndrome = torch.tensor([]).to(soft_output.device)
        for c in range(0, code.m):  # for each check node
            variable_nodes = []
            for v in range(0, code.n):
                if code.H[c, v] == 1:
                    variable_nodes.append(v)
            temp = soft_output[variable_nodes]
            (temp1, min_ind_temp2) = torch.min(torch.abs(temp), 0)
            soft_syndrome = torch.cat((soft_syndrome, temp1), 0)
        soft_syndrome = torch.reshape(soft_syndrome, (code.m, soft_output.size(dim=1)))
        return soft_syndrome
