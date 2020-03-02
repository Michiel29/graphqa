import logging
import os
import warnings

from fairseq import metrics, utils
from fairseq.tasks import register_task, TASK_REGISTRY

from tasks import BaseTask
from utils.data_utils import CustomDictionary, EntityDictionary


logger = logging.getLogger(__name__)


@register_task('multi_task')
class MultiTask(BaseTask):
    def __init__(self, args, dictionary, entity_dictionary, tasks):
        super().__init__(args, dictionary, entity_dictionary)
        self.tasks = tasks

    @classmethod
    def setup_task(cls, args, **kwargs):
        dict_path = os.path.join(args.data_path, 'dict.txt')
        dictionary = CustomDictionary.load(dict_path)

        entity_dict_path = os.path.join(args.data_path, 'entity.dict.txt')
        entity_dictionary = EntityDictionary.load(entity_dict_path)

        logger.info('dictionary: {} types'.format(len(dictionary)))
        logger.info('entity dictionary: {} types'.format(len(entity_dictionary)))

        tasks = [TASK_REGISTRY[task_name](args, dictionary, entity_dictionary) for task_name in args.tasks]
        task = cls(args, dictionary, entity_dictionary, tasks)
        return task

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        for task in self.tasks:
            task.load_dataset(split=split, epoch=epoch, combine=combine, **kwargs)