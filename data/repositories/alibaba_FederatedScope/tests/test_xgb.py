# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.core.auxiliaries.worker_builder import get_client_cls, get_server_cls
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.core.configs.config import global_cfg
from federatedscope.core.auxiliaries.runner_builder import get_runner


class XGBTest(unittest.TestCase):
    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def set_config(self, cfg):
        backup_cfg = cfg.clone()

        import torch
        cfg.use_gpu = torch.cuda.is_available()

        cfg.federate.mode = 'standalone'
        cfg.federate.client_num = 2

        cfg.model.type = 'xgb_tree'
        cfg.model.lambda_ = 1
        cfg.model.gamma = 0
        cfg.model.num_of_trees = 5
        cfg.model.max_tree_depth = 3

        cfg.train.optimizer.bin_num = 1000
        cfg.train.optimizer.eta = 0.5

        cfg.data.root = 'test_data/'
        cfg.data.type = 'adult'
        cfg.data.size = 2000

        cfg.dataloader.type = 'raw'

        cfg.criterion.type = 'CrossEntropyLoss'

        cfg.vertical.use = True
        cfg.vertical.xgb_use_bin = False
        cfg.vertical.dims = [5, 10]
        cfg.vertical.algo = 'xgb'

        cfg.trainer.type = 'verticaltrainer'
        cfg.eval.freq = 5
        cfg.eval.best_res_update_round_wise_key = "test_loss"

        return backup_cfg

    def test_XGBFL(self):
        init_cfg = global_cfg.clone()
        backup_cfg = self.set_config(init_cfg)
        setup_seed(init_cfg.seed)
        update_logger(init_cfg, True)

        data, modified_config = get_data(init_cfg.clone())
        init_cfg.merge_from_other_cfg(modified_config)
        self.assertIsNotNone(data)

        Fed_runner = get_runner(data=data,
                                server_class=get_server_cls(init_cfg),
                                client_class=get_client_cls(init_cfg),
                                config=init_cfg.clone())
        self.assertIsNotNone(Fed_runner)
        test_results = Fed_runner.run()
        init_cfg.merge_from_other_cfg(backup_cfg)
        print(test_results)
        self.assertGreater(test_results['server_global_eval']['test_acc'], 0.7)


if __name__ == '__main__':
    unittest.main()
