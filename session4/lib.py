import json
import pickle
import numpy as np
import os
import collections
import torch
import inspect
import signal
import argparse
import sys
from IPython import embed


class SumTingWongException(Exception):
    pass


def argparsify(f, test=None):
    args, _, _, defaults = inspect.getargspec(f)
    assert(len(args) == len(defaults))
    parser = argparse.ArgumentParser()
    i = 0
    for arg in args:
        argtype = type(defaults[i])
        if argtype == bool:     # convert to action
            if defaults[i] == False:
                action="store_true"
            else:
                action="store_false"
            parser.add_argument("-%s" % arg, "--%s" % arg, action=action, default=defaults[i])
        else:
            parser.add_argument("-%s"%arg, "--%s"%arg, type=type(defaults[i]))
        i += 1
    if test is not None:
        par = parser.parse_args([test])
    else:
        par = parser.parse_args()
    kwargs = {}
    for arg in args:
        if getattr(par, arg) is not None:
            kwargs[arg] = getattr(par, arg)
    return kwargs


def argprun(f, sigint_shell=True, **kwargs):   # command line overrides kwargs
    """ use this to enable command-line access to kwargs of function (useful for main run methods) """
    def handler(sig, frame):
        # find the frame right under the argprun
        print("custom handler called")
        original_frame = frame
        current_frame = original_frame
        previous_frame = None
        stop = False
        while not stop and current_frame.f_back is not None:
            previous_frame = current_frame
            current_frame = current_frame.f_back
            if "_FRAME_LEVEL" in current_frame.f_locals \
                and current_frame.f_locals["_FRAME_LEVEL"] == "ARGPRUN":
                stop = True
        if stop:    # argprun frame found
            __toexposelocals = previous_frame.f_locals     # f-level frame locals
            class L(object):
                pass
            l = L()
            for k, v in __toexposelocals.items():
                setattr(l, k, v)
            stopprompt = False
            while not stopprompt:
                whattodo = raw_input("(s)hell, (k)ill\n>>")
                if whattodo == "s":
                    embed()
                elif whattodo == "k":
                    "Killing"
                    sys.exit()
                else:
                    stopprompt = True

    if sigint_shell:
        _FRAME_LEVEL="ARGPRUN"
        prevhandler = signal.signal(signal.SIGINT, handler)
    try:
        f_args = argparsify(f)
        for k, v in kwargs.items():
            if k not in f_args:
                f_args[k] = v
        f(**f_args)

    except KeyboardInterrupt:
        print("Interrupted by Keyboard")


class StringMatrix():
    protectedwords = ["<MASK>", "<RARE>", "<START>", "<END>"]

    def __init__(self, maxlen=None, freqcutoff=0, topnwords=None, indicate_start_end=False, indicate_start=False, indicate_end=False):
        self._strings = []
        self._wordcounts_original = dict(zip(self.protectedwords, [0] * len(self.protectedwords)))
        self._dictionary = dict(zip(self.protectedwords, range(len(self.protectedwords))))
        self._dictionary_external = False
        self._rd = None
        self._next_available_id = len(self._dictionary)
        self._maxlen = 0
        self._matrix = None
        self._max_allowable_length = maxlen
        self._rarefreq = freqcutoff
        self._topnwords = topnwords
        self._indic_e, self._indic_s = False, False
        if indicate_start_end:
            self._indic_s, self._indic_e = True, True
        if indicate_start:
            self._indic_s = indicate_start
        if indicate_end:
            self._indic_e = indicate_end
        self._rarewords = set()
        self.tokenize = lambda x: x.lower().strip().split()
        self._cache_p = None

    def clone(self):
        n = StringMatrix()
        n.tokenize = self.tokenize
        if self._matrix is not None:
            n._matrix = self._matrix.copy()
            n._dictionary = self._dictionary.copy()
            n._rd = self._rd.copy()

        n._strings = self._strings
        return n

    def __len__(self):
        if self._matrix is None:
            return len(self._strings)
        else:
            return self.matrix.shape[0]

    def __getitem__(self, item, *args):
        if self._matrix is None:
            return self._strings[item]
        else:
            ret = self.matrix[item]
            if len(args) == 1:
                ret = ret[args[0]]
            ret = self.pp(ret)
            return ret

    @property
    def numwords(self):
        return len(self._dictionary)

    @property
    def numrare(self):
        return len(self._rarewords)

    @property
    def matrix(self):
        if self._matrix is None:
            raise Exception("finalize first")
        return self._matrix

    @property
    def D(self):
        return self._dictionary

    def set_dictionary(self, d):
        """ dictionary set in this way is not allowed to grow,
        tokens missing from provided dictionary will be replaced with <RARE>
        provided dictionary must contain <RARE> if missing tokens are to be supported"""
        print("setting dictionary")
        self._dictionary_external = True
        self._dictionary = {}
        self._dictionary.update(d)
        self._next_available_id = max(self._dictionary.values()) + 1
        self._wordcounts_original = dict(zip(list(self._dictionary.keys()), [0]*len(self._dictionary)))
        self._rd = {v: k for k, v in self._dictionary.items()}

    @property
    def RD(self):
        return self._rd

    def d(self, x):
        return self._dictionary[x]

    def rd(self, x):
        return self._rd[x]

    def pp(self, matorvec):
        def pp_vec(vec):
            return " ".join([self.rd(x) if x in self._rd else "<UNK>" for x in vec if x != self.d("<MASK>")])
        ret = []
        if matorvec.ndim == 2:
            for vec in matorvec:
                ret.append(pp_vec(vec))
        else:
            return pp_vec(matorvec)
        return ret

    def add(self, x):
        tokens = self.tokenize(x)
        tokens = tokens[:self._max_allowable_length]
        if self._indic_s is not False and self._indic_s is not None:
            indic_s_sym = "<START>" if not isstring(self._indic_s) else self._indic_s
            tokens = [indic_s_sym] + tokens
        if self._indic_e is not False and self._indic_e is not None:
            indic_e_sym = "<END>" if not isstring(self._indic_e) else self._indic_e
            tokens = tokens + [indic_e_sym]
        self._maxlen = max(self._maxlen, len(tokens))
        tokenidxs = []
        for token in tokens:
            if token not in self._dictionary:
                if not self._dictionary_external:
                    self._dictionary[token] = self._next_available_id
                    self._next_available_id += 1
                    self._wordcounts_original[token] = 0
                else:
                    assert("<RARE>" in self._dictionary)
                    token = "<RARE>"    # replace tokens missing from external D with <RARE>
            self._wordcounts_original[token] += 1
            tokenidxs.append(self._dictionary[token])
        self._strings.append(tokenidxs)
        return len(self._strings)-1

    def finalize(self):
        ret = np.zeros((len(self._strings), self._maxlen), dtype="int64")
        for i, string in enumerate(self._strings):
            ret[i, :len(string)] = string
        self._matrix = ret
        self._do_rare_sorted()
        self._rd = {v: k for k, v in self._dictionary.items()}
        self._strings = None

    def _do_rare_sorted(self):
        """ if dictionary is not external, sorts dictionary by counts and applies rare frequency and dictionary is changed """
        if not self._dictionary_external:
            sortedwordidxs = [self.d(x) for x in self.protectedwords] + \
                             ([self.d(x) for x, y
                              in sorted(self._wordcounts_original.items(), key=lambda (x, y): y, reverse=True)
                              if y >= self._rarefreq and x not in self.protectedwords][:self._topnwords])
            transdic = zip(sortedwordidxs, range(len(sortedwordidxs)))
            transdic = dict(transdic)
            self._rarewords = {x for x in self._dictionary.keys() if self.d(x) not in transdic}
            rarewords = {self.d(x) for x in self._rarewords}
            self._numrare = len(rarewords)
            transdic.update(dict(zip(rarewords, [self.d("<RARE>")]*len(rarewords))))
            # translate matrix
            self._matrix = np.vectorize(lambda x: transdic[x])(self._matrix)
            # change dictionary
            self._dictionary = {k: transdic[v] for k, v in self._dictionary.items() if self.d(k) in sortedwordidxs}

    def save(self, p):
        pickle.dump(self, open(p, "w"))

    @staticmethod
    def load(p):
        if os.path.isfile(p):
            return pickle.load(open(p))
        else:
            return None


def isstring(x):
    return isinstance(x, basestring)


def issequence(x):
    return isinstance(x, collections.Sequence) and not isinstance(x, basestring)


def iscuda(x):
    if isinstance(x, torch.nn.Module):
        params = list(x.parameters())
        return params[0].is_cuda
    else:
        raise SumTingWongException("unsupported type")


def load_jsons(datap="resources/qald_combined.json",
               relp="resources/nrels.json",
               mode="flat"):
    """ relp: file must contain dictionary mapping relation ids (ints) to lists of words (strings)"""
    """ mode: "flat", "slotptr" """
    print("loading jsons")

    data = json.load(open(datap))
    rels = json.load(open(relp))

    print("jsons loaded")

    print("extracting data")
    questions = []
    goldchains = []
    badchains = []
    for dataitem in data:
        questions.append(dataitem["parsed-data"]["corrected_question"])
        goldchain = []
        if not dataitem["parsed-data"]["path_id"] == [-1]:
            for x in dataitem["parsed-data"]["path_id"]:
                goldchain += [x[0], int(x[1:])]
        goldchains.append(goldchain)
        badchainses = []
        goldfound = False
        for badchain in dataitem["uri"]["hop-1-properties"] + dataitem["uri"]["hop-2-properties"]:
            if goldchain == badchain:
                goldfound = True
            else:
                if len(badchain) == 2:
                    badchain += [-1, -1]
                badchainses.append(badchain)
        badchains.append(badchainses)

    print("extracted data")

    print("mode: {}".format(mode))
    print("flattening")

    def flatten_chain(chainspec):
        if len(chainspec) == 0:
            ret = u"EMPTYEMPTYEMPTY"
        else:
            flatchainspec = []
            for x in chainspec:
                if x in (u"+", u"-"):
                    flatchainspec.append(x)
                elif x > -1:
                    relwords = rels[str(x)]
                    flatchainspec += relwords
                elif x == -1:
                    pass
                else:
                    raise SumTingWongException("unexpected symbol in chain")
            ret = u" ".join(flatchainspec).lower()
        return ret

    goldchainids = []
    badchainsids = []

    uniquechainids = {}

    qsm = StringMatrix()
    csm = StringMatrix()
    csm.tokenize = lambda x: x.strip().split()

    def get_ensure_chainid(flatchain):
        if flatchain not in uniquechainids:
            uniquechainids[flatchain] = len(uniquechainids)
            csm.add(flatchain)
            assert(len(csm) == len(uniquechainids))
        return uniquechainids[flatchain]

    eid = 0
    numchains = 0
    for question, goldchain, badchainses in zip(questions, goldchains, badchains):
        qsm.add(question)
        # flatten gold chain
        flatgoldchain = flatten_chain(goldchain)
        chainid = get_ensure_chainid(flatgoldchain)
        goldchainids.append(chainid)
        badchainsids.append([])
        numchains += 1
        for badchain in badchainses:
            flatbadchain = flatten_chain(badchain)
            chainid = get_ensure_chainid(flatbadchain)
            badchainsids[eid].append(chainid)
            numchains += 1
        eid += 1
        # print("{}".format(eid))

    assert(len(badchainsids) == len(questions))

    print("{} unique chains from {} total".format(len(csm), numchains))
    qsm.finalize()
    csm.finalize()
    print("flattened")
    csm.tokenize = None
    return qsm, csm, goldchainids, badchainsids


class RankingComputer(object):
    """ computes rankings based on ranking model for full validation/test ranking
        provides separate loss objects to put into lossarray
    """

    def __init__(self, scoremodel, eids, ldata, rdata, eid2rid_gold, eid2rid_neg):
        self.scoremodel = scoremodel
        self.eids = eids
        self.ldata = ldata if issequence(ldata) else (ldata,)     # already shuffled
        self.rdata = rdata if issequence(rdata) else (rdata,)     # indexed by eid space
        self.eid2rid_neg = eid2rid_neg      # indexed by eid space
        self.eid2rid_gold = eid2rid_gold    # indexed by eid space

    def compute(self, *metrics):        # compute given metrics for all given data
        self.scoremodel.train(False)
        rankings = self.compute_rankings(self.eids)
        metricnumbers = []
        for i, metric in enumerate(metrics):
            metricnumbers.append(metric.compute(rankings))
        # TODO
        return metricnumbers

    def compute_rankings(self, eids):
        cuda = iscuda(self.scoremodel)
        # get all pairs to score
        current_batch = []
        # given questions are already shuffled --> just traverse
        for eid, ldata_id in zip(list(eids), range(len(self.eids))):
            rdata = []
            rids = [self.eid2rid_gold[eid]] + list(set(self.eid2rid_neg[eid]) - {self.eid2rid_gold[eid],})
            ldata = [ldat[ldata_id][np.newaxis, ...].repeat(len(rids), axis=0)
                          for ldat in self.ldata]
            trueornot = [0] * len(rids)
            trueornot[0] = 1
            for rid in rids:
                right_data = tuple([rdat[rid] for rdat in self.rdata])
                rdata.append(right_data)
            rdata = zip(*rdata)
            ldata = [torch.tensor(ldat) for ldat in ldata]
            rdata = [torch.tensor(np.stack(posdata_e)) for posdata_e in rdata]
            scores = self.scoremodel(ldata[0], rdata[0])
            _scores = list(scores.detach().cpu().numpy())
            ranking = sorted(zip(_scores, rids, trueornot), key=lambda x: x[0], reverse=True)
            current_batch.append((eid, ranking))
        return current_batch


class RecallAt(object):
    def __init__(self, k, totaltrue=None, **kw):
        super(RecallAt, self).__init__(**kw)
        self.k = k
        self.totaltrue = totaltrue

    def compute(self, rankings, **kw):
        # list or (eid, lid, ranking)
        # ranking is a list of triples (_scores, rids, trueornot)
        ys = []
        for _, ranking in rankings:
            topktrue = 0.
            totaltrue = 0.
            for i in range(len(ranking)):
                _, _, trueornot = ranking[i]
                if i < self.k:
                    topktrue += trueornot
                else:
                    if self.totaltrue is not None:
                        totaltrue = self.totaltrue
                        break
                if trueornot == 1:
                    totaltrue += 1.
            topktrue = topktrue / totaltrue
            ys.append(topktrue)
        ys = np.stack(ys)
        return ys


def datasplit(npmats, splits=(80, 20), random=True):
    splits = np.round(len(npmats[0]) * np.cumsum(splits) / sum(splits)).astype("int32")

    whatsplit = np.zeros((len(npmats[0]),), dtype="int64")
    for i in range(1, len(splits)):
        a, b = splits[i-1], splits[i]
        whatsplit[a:b] = i

    if random is not False and random is not None:
        if isinstance(random, int):
            np.random.seed(random)
            random = True

        if random is True:
            np.random.shuffle(whatsplit)

    ret = []
    for i in range(0, len(splits)):
        splitmats = [npmat[whatsplit == i] for npmat in npmats]
        ret.append(splitmats)
    return ret



# SEQUENCE PACKING AND UNPACKING
def seq_pack(x, mask):  # mask: (batsize, seqlen)
    """ given N-dim sequence "x" (N>=2), and 2D mask (batsize, seqlen)
        returns packed sequence (sorted) and indexes to un-sort (also used by seq_unpack) """
    x = x.float()
    mask = mask.float()
    # 1. get lengths
    lens = torch.sum(mask.float(), 1)
    # 2. sort by length
    assert(lens.dim() == 1)
    _, sortidxs = torch.sort(lens, descending=True)
    unsorter = torch.zeros(sortidxs.size(), dtype=torch.int64).to(sortidxs.device)
    # print ("test unsorter")
    # print (unsorter)
    unsorter.data.scatter_(0, sortidxs.data,
                           torch.arange(0, len(unsorter)).to(sortidxs.device).long())
    # 3. pack
    sortedseq = torch.index_select(x, 0, sortidxs)
    sortedmsk = torch.index_select(mask, 0, sortidxs)
    sortedlens = sortedmsk.long().sum(1)
    sortedlens = list(sortedlens.numpy())
    packedseq = torch.nn.utils.rnn.pack_padded_sequence(sortedseq, sortedlens, batch_first=True)
    return packedseq, unsorter


def seq_unpack(x, order, padding_value=0):
    """ given packed sequence "x" and the un-sorter "order",
        returns padded sequence (un-sorted by "order") and a binary 2D mask (batsize, seqlen),
            where padded sequence is padded with "padding_value" """
    unpacked, lens = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True, padding_value=padding_value)
    mask = torch.zeros(len(lens), max(lens), dtype=torch.int64).to(unpacked.device)
    for i, l in enumerate(lens):
        mask.data[i, :l] = 1
    out = torch.index_select(unpacked, 0, order)        # same as: unpacked[order]
    outmask = torch.index_select(mask, 0, order)        # same as: mask[order]
    return out, outmask


class BestWriter(object):
    def __init__(self, qsm, csm, p=None, **kw):
        super(BestWriter, self).__init__()
        self.qsm = qsm
        self.csm = csm
        self.p = p

    def compute(self, rankings, **kw):
        ds = []
        with open(self.p, "w") as f:
            for eid, ranking in rankings:
                question_of_example = self.qsm[eid]
                best_scored_chain_of_example = self.csm[ranking[0][1]]
                est_truth_of_best_scored_chain = str(bool(ranking[0][2])).lower()
                number_of_chains_in_ranking_for_example = len(ranking)
                d = {"eid": eid,
                     "question": question_of_example,
                     "best_chain": best_scored_chain_of_example,
                     "best_chain_ass_truth": est_truth_of_best_scored_chain,
                     "num_chains": number_of_chains_in_ranking_for_example}
                ds.append(d)
            json.dump(ds, f, indent=2, sort_keys=True)
        return 0
