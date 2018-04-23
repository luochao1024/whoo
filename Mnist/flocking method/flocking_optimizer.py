"""Wrapper optimizer for Elastic Average SGD """
import tensorflow as tf
from tensorflow.python.training import optimizer

GLOBAL_VARIABLES_NAME = 'global_variables'
RECORD_LOCAL_VARIABLES_NAME = 'record_local_variables'


class FlockingCustomGetter(object):
    '''custom_getter is used with tf.get_variable instead of tf.Variablel.
    This custom_getter is used to:
    1. place the trainable variables(local_variables) to tower_device(gpu)
    2. generate global_variables(a copy of_local variables) and place them on cpu, 
     which are untrainable are can be assessed by other tower to calculate the flocking_magnitude
    3. generate record_local_variables which are untrainable and are placed on tower_device(gpu).
     The number of record_local_variables is num_towers * local_variables. They are 
     used to record the global_variables of all towers(including the current tower).
     
    For example,
    flocking_custom_getter = FlockingCustomGetter(num_towers, tower_device)
    with tf.variable_scope("tower_1", custom_getter=flocking_custom_getter):
        weights = tf.get_variable(initializer=tf.truncated_normal([28*28, 30], stddev=1.0/28), name="weights")
        biases = tf.get_variable(initializer=tf.zeros([30]), name="biases")
    '''
    
    def __init__(self, num_towers, tower_index):
        '''create a new 'FlockingCustomGetter'.'''
        self.num_towers = num_towers
        self.tower_index = tower_index
        
        #a list store all the local_variables in the current tower.
        self.local_variables_list = []
        
        #a list store all the global_variables in the current tower.
        self.global_variables_list = [] 
        
        #a list store all the record_local_variables including the current tower.
        #it is a record of all trainable variables
        self.record_local_variables_list= [[] for _ in range(num_towers)] 
         
    def __call__(self, getter, name, trainable, collections, *args, **kwargs):
        if trainable:
            with tf.device('/cpu:0'):
                local_variables = getter(
                    name, 
                    trainable=True, 
                    collections=[tf.GraphKeys.LOCAL_VARIABLES],
                    *args,
                    **kwargs)
            self.local_variables_list.append(local_variables)    
            
            with tf.device('cpu/:0'):
                global_variables = tf.Variable(name='%s/%s' % (GLOBAL_VARIABLES_NAME, name),
                                               initial_value=local_variables.initialized_value(),
                                               trainable=False,
                                               collections=[tf.GraphKeys.GLOBAL_VARIABLES])
            self.global_variables_list.append(global_variables)
            
            with tf.device('/cpu:0'):
                for i in range(self.num_towers):
                    rlv = tf.Variable(name='%s/%s_%d' % (GLOBAL_VARIABLES_NAME, name, i),
                                      initial_value=local_variables.initialized_value(),
                                      trainable=False,
                                      collections=[tf.GraphKeys.LOCAL_VARIABLES])            
                    self.record_local_variables_list[i].append(rlv)
                    
            return local_variables
        else:
            return getter(name, trainable, collections, *args, **kwargs)


class FlockingOptimizer(optimizer.Optimizer):
    '''Wrapper optimizer that implements the Flocking SGD algorithm. This is an async
    optimizer. During the training, each tower will copy all global_variables and stores
    them at record_local_variables. And then each tower will calculate the average of 
    record_local_variables based on the 0 dimension which is actually the center for 
    each variable from different tower. Each tower then calcualte the distance between
    local_variables and the average of record_local_variables. The distance is used to
    calculate the flocking_magnitude to update both the local_variables and global_variables. 
    Also, after each tower updates the local_variables, the tower's own local_step will
    be incremented by 1; after corresponding global_variables are updated, global_step will
    be incremented by 1
    '''
    
    def __init__(self,
                 opt,
                 num_towers,
                 tower_index,
                 flocking_custom_getter_list,
                 attraction = 1.0,
                 repulsion = 3.0,
                 dis = 0.005,
                 use_locking=True,
                 name='FlockingOptimizer'):
        super(FlockingOptimizer, self).__init__(use_locking, name)
        self._opt = opt
        self._num_towers = num_towers
        self._attra = attraction
        self._repul = repulsion
        self._dis = dis
        self._global_vars_all_towers = [fcg.global_variables_list for fcg in 
                                       flocking_custom_getter_list]
        self._local_vars_list = [var for var in 
                                 flocking_custom_getter_list[tower_index].local_variables_list]
        self._record_local_variables = flocking_custom_getter_list[tower_index].record_local_variables_list
        self._slope = self._repul/self._dis
        self.tower_index = tower_index
        
        with tf.variable_scope('local_step_%d' % tower_index):
            self._local_step = tf.get_variable(initializer=0,
                                               trainable=False,
                                               collections=[tf.GraphKeys.LOCAL_VARIABLES],
                                               name='')
        self._opt._prepare()
        
    def compute_gradients(self,
                          loss,
                          gate_gradients=optimizer.Optimizer.GATE_OP,
                          aggregation_method=None,
                          colocate_gradients_with_ops=False,
                          grad_loss=None):
        '''compute gradients of 'loss' for variables in 'var_list', it returns 
        a list of (gradient, variable) pairs where "gradient" is the gradient for
        "variable". Note that "gradient" can be a 'Tensor', an 'IndexedSlices'.or
        'None' if there is no gradient for the given variable.
        
        Args:
        loss: A Tensor containing the value to minimize.
        tower_index: the tower to update its local_variables.
        gate_gradients: How to gate the computation of gradients.  Can be
          `GATE_NONE`, `GATE_OP`, or `GATE_GRAPH`.
        aggregation_method: Specifies the method used to combine gradient terms.
          Valid values are defined in the class `AggregationMethod`.
        colocate_gradients_with_ops: If True, try colocating gradients with
          the corresponding op.
        grad_loss: Optional. A `Tensor` holding the gradient computed for `loss`.
      Returns:
        A list of (gradient, variable) pairs. Variable is always present, but
        gradient can be `None`.
      Raises:
        TypeError: If `var_list` contains anything else than `Variable` objects.
        ValueError: If some arguments are invalid.
        '''
        return self._opt.compute_gradients(loss, self._local_vars_list, gate_gradients,
                                               aggregation_method,
                                               colocate_gradients_with_ops,
                                               grad_loss)
    
    def apply_gradients_and_flocking(self, tower_index, grads_and_vars, global_step=None, name=None):
        '''apply flocking_magnitude and gradients for local_variables of current tower'''
        def _apply_flocking():
            local_var_list = [v for g, v in grads_and_vars]
            tower_device = local_var_list[0].device
            record_local_variables_update=[] # a list to store ops
            with tf.device(tower_device):
                for rlv_one_tower, gv_one_tower in zip(self._record_local_variables, 
                                                       self._global_vars_all_towers):
                    for rlv, gv in zip(rlv_one_tower, gv_one_tower):
                        record_local_variables_update.append(rlv.assign(gv))
                        
                with tf.control_dependencies(record_local_variables_update):
                    same_variables_from_different_towers = zip(*self._record_local_variables)
                    average_variables = [tf.add_n(s_v)/self._num_towers for 
                                                s_v in same_variables_from_different_towers]
                    
                    distance = [tf.subtract(local, average) for local, average in 
                                zip(local_var_list, average_variables)]          
                    flocking_function = [tf.minimum(self._slope*tf.abs(dis)-self._repul+self._attra, 1.0) 
                                         for dis in distance]
                    flocking_magnitudes=[tf.multiply(dis, f) for dis, f in
                                         zip(distance, flocking_function)]
                    local_var_update_ops = [tf.subtract(var, f_mag) for var, f_mag in
                                            zip(local_var_list, flocking_magnitudes)]
            with tf.control_dependencies(local_var_update_ops):
                with tf.device('/cpu:0'):
                    global_var_update_ops=[g_v.assign(l_v) for g_v, l_v in
                                           zip(self._global_vars_all_towers[tower_index],
                                               local_var_list)]
            if global_step:
                with tf.colocate_with(global_step):
                    local_var_update_ops.append(tf.assign_add(global_step, 1))
                                                 
            var_update = tf.group(*(local_var_update_ops), *(global_var_update_ops))
            return var_update
        
        #apply gradients first    
        apply_gradients_ops = self._opt.apply_gradients(grads_and_vars)
        #and then apply flocking_magnitude
        with tf.control_dependencies([apply_gradients_ops]):
            update = _apply_flocking()
        return update
    