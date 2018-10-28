import tensorflow as tf
import tensorflow.contrib.slim as slim
import os

class RestoreCheckpointHook(tf.train.SessionRunHook):


    def __init__(self,
        checkpoint_path,
        exclude_scope_patterns,
        include_scope_patterns
        ):

        tf.logging.info("Create RestoreCheckpointHook.")

        if checkpoint_path :

            if tf.gfile.IsDirectory(checkpoint_path):
                self.checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
            else:
                self.checkpoint_path = checkpoint_path


        #super(IteratorInitializerHook, self).__init__()
        # self.checkpoint_path = checkpoint_path

        self.exclude_scope_patterns = None if (not exclude_scope_patterns) else [scope.strip() for scope in exclude_scope_patterns.split(",")]
        self.include_scope_patterns = None if (not include_scope_patterns) else include_scope_patterns.split(',')



    def begin(self):
        # You can add ops to the graph here.
        print('Before starting the session--create saver to restore.')

        # 1. Create saver

        #exclusions = []
        #if self.checkpoint_exclude_scopes:
        # exclusions = [scope.strip()
        # for scope in self.checkpoint_exclude_scopes.split(',')]
        #
        #variables_to_restore = []
        #for var in slim.get_model_variables(): #tf.global_variables():
        # excluded = False
        # for exclusion in exclusions:
        # if var.op.name.startswith(exclusion):
        # excluded = True
        # break
        # if not excluded:
        # variables_to_restore.append(var)
        #inclusions
        #[var for var in tf.trainable_variables() if var.op.name.startswith('InceptionResnetV1')]

        variables_to_restore = tf.contrib.framework.filter_variables(
        slim.get_model_variables(),
        include_patterns=self.include_scope_patterns, # ['Conv'],
        exclude_patterns=self.exclude_scope_patterns, # ['biases', 'Logits'],

        # If True (default), performs re.search to find matches
        # (i.e. pattern can match any substring of the variable name).
        # If False, performs re.match (i.e. regexp should match from the beginning of the variable name).
        reg_search = True
        )
        self.saver = tf.train.Saver(variables_to_restore)


    def after_create_session(self, session, coord):
        # When this is called, the graph is finalized and
        # ops can no longer be added to the graph.

        print('ckpt restoring variables...')


        # session.run(self.iterator_init_op)

        tf.logging.info('Fine-tuning from %s' % self.checkpoint_path)
        self.saver.restore(session, os.path.expanduser(self.checkpoint_path))
        tf.logging.info('End fineturn from %s' % self.checkpoint_path)

    def before_run(self, run_context):
        #print('Before calling session.run().')
        return None #SessionRunArgs(self.your_tensor)

    def after_run(self, run_context, run_values):
        #print('Done running one step. The value of my tensor: %s', run_values.results)
        #if you-need-to-stop-loop:
        # run_context.request_stop()
        pass


    def end(self, session):
        print('Done with the session--train.')
        pass


# 自定义钩子
class IteratorInitHook(tf.train.SessionRunHook):



    def after_create_session(self, session, coord):
        print("ceated init session")
        print("init iterator--------------")
        session.run(self.iterator_init_op)