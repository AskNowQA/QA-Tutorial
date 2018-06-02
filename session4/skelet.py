from lib import *
import torch
from torch.utils.data import TensorDataset, DataLoader
import random


class RankModel(torch.nn.Module):
    def __init__(self, scoremodel, margin=1.):
        super(RankModel, self).__init__()
        self.scoremodel = scoremodel
        self.margin = margin

    def compute_loss(self, psim, nsim):
        '''
        :param psim: the score between the question and the correct chain, shape = (batch_size,)
        :param nsim: the score between the question and the bad chain, shape = (batch_size,)
        :return: array of losses, shape (batch_size,)
        '''
        # TODO: implement this !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #
        # -----------------------------------------------------------------
        pass

    def forward(self, question, good_chain, bad_chain):
        psim = self.scoremodel(question, good_chain)
        nsim = self.scoremodel(question, bad_chain)
        loss = self.compute_loss(psim, nsim)
        return loss


class ScoreModel(torch.nn.Module):
    # TODO: implement this !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #
    # ---------------------------------------------------------------------
    pass

def run(lr=0.001,
        batch_size=10,
        epochs=100,
        margin=1.,
        embdim=50,
        encdim=50,
        ):
    print("loading data")
    qsm, csm, goldchainids, badchainids = load_jsons()
    print("data loaded")
    print("Question: \t{}".format(qsm[3]))
    print("Query:\t\t{}".format(csm[goldchainids[3]]))
    eids = np.arange(0, len(qsm))

    data = [qsm.matrix, eids]
    traindata, validdata, testdata = datasplit(data, splits=(7, 1, 2), random=False)
    print("Number of training examples: {}".format(len(traindata[0])))

    tensordataset = torch.utils.data.TensorDataset(*[torch.tensor(x) for x in traindata])
    dataloader = torch.utils.data.DataLoader(tensordataset, batch_size=batch_size, shuffle=True)

    score_model = ScoreModel(len(qsm.D), embdim, len(csm.D), embdim, encdim)
    rank_model = RankModel(scoremodel=score_model, margin=margin)

    optimizer = torch.optim.Adam(score_model.parameters(), lr=lr)

    print(score_model)

    for i in range(epochs):
        print("Epoch {}".format(i))
        for questions_batch, eids_batch in dataloader:
            rank_model.zero_grad()
            # region materialize chains
            good_chains = np.zeros((len(eids_batch),) + csm.matrix.shape[1:], dtype="int64")
            bad_chains = np.zeros((len(eids_batch),) + csm.matrix.shape[1:], dtype="int64")

            for i, eid in enumerate(eids_batch.numpy()):
                good_chains[i, ...] = csm.matrix[goldchainids[eid]]
                badcidses = badchainids[eid]
                if len(badcidses) == 0:
                    badcid = random.randint(0, len(csm) - 1)
                else:
                    badcid = random.sample(badcidses, 1)[0]
                bad_chains[i, ...] = csm.matrix[badcid]

            good_chains = torch.tensor(good_chains).to(eids_batch.device)
            bad_chains = torch.tensor(bad_chains).to(eids_batch.device)
            # endregion

            loss = rank_model(questions_batch, good_chains, bad_chains).mean()
            print(loss.item())

            loss.backward()

            optimizer.step()
        print("Validating")
        valid_numbers = test_model(score_model, validdata, qsm, csm, goldchainids, badchainids, "valid")
        print("Recall@1: {:.3f} \t Recall@5: {:.3f} \t ({} examples)".format(*(valid_numbers + [len(validdata[0])])))
    print("Testing")
    test_numbers = test_model(score_model, testdata, qsm, csm, goldchainids, badchainids, "test")
    print("Recall@1: {:.3f} \t Recall@5: {:.3f} \t ({} examples)".format(*(test_numbers + [len(testdata[0])])))


def test_model(scoremodel, _data, qsm, csm, good_chain_ids, bad_chain_ids, prefix):     # TODO
    rankcomp = RankingComputer(scoremodel, _data[1], _data[0], csm.matrix, good_chain_ids, bad_chain_ids)
    rankmetrics = rankcomp.compute(RecallAt(1, totaltrue=1),
                                   RecallAt(5, totaltrue=1),
                                   BestWriter(qsm, csm, p="{}.out".format(prefix)))
    ret = [np.asarray(rankmetric).mean() for rankmetric in rankmetrics]
    return ret

if __name__ == "__main__":
    argprun(run)