# BCE loss function
def binary_cross_entropy(y_hat,y,weights):
    # y is the output after sigmoid
    losses = (y_hat+0.000000001).log()*y + (1-y_hat+0.000000001).log()*(1-y) 
    losses = losses*weights
    losses = -1*losses
    return(losses.sum(axis=1))

# get data batches
def get_batch(batch, ctx):
    """Return features and labels on ctx."""
    features, labels = batch
    if labels.dtype != features.dtype:
        labels = labels.astype(features.dtype)
    return (gutils.split_and_load(features, ctx),
            gutils.split_and_load(labels, ctx), features.shape[0])

# calculate F1 score
def F1(p,r):
    return((2*p*r)/(p+r))

# evaluate the model and save results
def evaluate(net):
    print('evaluating...')
    metrics = {}
    
    for i,batch in tqdm(enumerate(iter_test)):
        Xs,ys,batch_size = get_batch(batch, d2l.try_all_gpus())
        y_hats = [net(X) for X in Xs]
        y_hats = y_hats[0]
        if i == 0:
            predict_prob = y_hats
        else:
            predict_prob = nd.concat(predict_prob,y_hats,dim=0)
        
    # macro-F1
    prediction = predict_prob>prob_thr 
    common_pos = (labels_test*prediction).sum(axis=0,keepdims=False)+1
    precision = common_pos / (prediction.sum(axis=0,keepdims=False)+2)
    recall = common_pos / (labels_test.sum(axis=0,keepdims=False)+2)
    macro_F1 = F1(precision.mean(), recall.mean())
    macro_F1_1 = F1(precision[0:c1].mean(), recall[0:c1].mean())
    macro_F1_2 = F1(precision[c1:(c1+c2)].mean(), recall[c1:(c1+c2)].mean())
    macro_F1_3 = F1(precision[(c1+c2):].mean(), recall[(c1+c2):].mean())
    
    metrics['macro_F1'] = macro_F1.asscalar()
    metrics['macro_F1_1'] = macro_F1_1.asscalar()
    metrics['macro_F1_2'] = macro_F1_2.asscalar()
    metrics['macro_F1_3'] = macro_F1_3.asscalar()
    
    # micro-F1
    total_common_pos = (labels_test*prediction).sum()
    micro_precision = total_common_pos/prediction.sum()
    micro_recall = total_common_pos/labels_test.sum()
    micro_F1 = F1(micro_precision, micro_recall)
    
    total_common_pos_1 = (labels_test*prediction)[:,0:c1].sum()
    micro_precision_1 = total_common_pos_1/prediction[:,0:c1].sum()
    micro_recall_1 = total_common_pos_1/labels_test[:,0:c1].sum()
    micro_F1_1 = F1(micro_precision_1, micro_recall_1)
    
    total_common_pos_2 = (labels_test*prediction)[:,c1:(c1+c2)].sum()
    micro_precision_2 = total_common_pos_2/prediction[:,c1:(c1+c2)].sum()
    micro_recall_2 = total_common_pos_2/labels_test[:,c1:(c1+c2)].sum()
    micro_F1_2 = F1(micro_precision_2, micro_recall_2)
    
    total_common_pos_3 = (labels_test*prediction)[:,(c1+c2):].sum()
    micro_precision_3 = total_common_pos_3/prediction[:,(c1+c2):].sum()
    micro_recall_3 = total_common_pos_3/labels_test[:,(c1+c2):].sum()
    micro_F1_3 = F1(micro_precision_3, micro_recall_3)
    
    metrics['micro_F1'] = micro_F1.asscalar()
    metrics['micro_F1_1'] = micro_F1_1.asscalar()
    metrics['micro_F1_2'] = micro_F1_2.asscalar()
    metrics['micro_F1_3'] = micro_F1_3.asscalar()
    
    # accuracy
    consistency = labels_test==prediction
    acc = (consistency.sum(axis=1,keepdims=False)==c1+c2+c3).sum()/labels_test.shape[0]
    acc1 = (consistency[:,0:c1].sum(axis=1,keepdims=False)==c1).sum()/labels_test.shape[0]
    acc2 = (consistency[:,c1:(c1+c2)].sum(axis=1,keepdims=False)==c2).sum()/labels_test.shape[0]
    acc3 = (consistency[:,(c1+c2):].sum(axis=1,keepdims=False)==c3).sum()/labels_test.shape[0]
    
    metrics['accuracy'] = acc.asscalar()
    metrics['accuracy_1'] = acc1.asscalar()
    metrics['accuracy_2'] = acc2.asscalar()
    metrics['accuracy_3'] = acc3.asscalar()
    
    return metrics

# initialize a log for storing evaluation results
def init_log():
    log = pd.DataFrame()
    log['epoch'] = None
    log['loss'] = None
    log['micro_F1'] = None
    log['micro_F1_1'] = None
    log['micro_F1_2'] = None
    log['micro_F1_3'] = None
    log['macro_F1'] = None
    log['macro_F1_1'] = None
    log['macro_F1_2'] = None
    log['macro_F1_3'] = None
    log['accuracy'] = None
    log['accuracy_1'] = None
    log['accuracy_2'] = None
    log['accuracy_3'] = None
    return log

# training function
def train(train_iter, test_iter, net, trainer, ctx, num_epochs):
    print('start training on ', ctx)
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    for epoch in range(num_epochs):
        train_l_sum, n = 0.0, 0
        for i,batch in tqdm(enumerate(train_iter)):
            Xs, ys, batch_size = get_batch(batch, ctx)
            ls = []
            with autograd.record():
                y_hats = [net(X) for X in Xs]

                # binary-cross-entropy loss
                ls11 = [loss1(y_hat[:,0:c1],y[:,0:c1],weights_1) for y_hat,y in zip(y_hats,ys)][0]
                ls12 = [loss1(y_hat[:,c1:(c1+c2)],y[:,c1:(c1+c2)],weights_2) for y_hat,y in zip(y_hats,ys)][0]
                ls13 = [loss1(y_hat[:,(c1+c2):(c1+c2+c3)],y[:,(c1+c2):(c1+c2+c3)],weights_3) for y_hat,y in zip(y_hats,ys)][0]
                ls = ls11+ ls12 + ls13 # sample weighting
                
            ls.backward()
            trainer.step(batch_size)
            train_l_sum += ls.sum().asscalar()
            n += batch_size
        metrics = evaluate(net)
        print('epoch %d, train loss %.4f, \n \
              micro-F1 %.4f, micro-F1-1 %.4f, micro-F1-2 %.4f, micro-F1-3 %.4f, \n \
              macro-F1 %.4f, macro-F1-1 %.4f, macro-F1-2 %.4f, macro-F1-3 %.4f, \n \
              accuracy %.4f, accuracy-1 %.4f, accuracy-2 %.4f, accuracy-3 %.4f'
              % (epoch + 1, train_l_sum / n, 
                 metrics['micro_F1'], metrics['micro_F1_1'], metrics['micro_F1_2'], metrics['micro_F1_3'],
                metrics['macro_F1'], metrics['macro_F1_1'], metrics['macro_F1_2'], metrics['macro_F1_3'],
                metrics['accuracy'], metrics['accuracy_1'], metrics['accuracy_2'], metrics['accuracy_3']))
        global log
        metrics['epoch'] = epoch+1
        metrics['loss'] = train_l_sum/n
        log = log.append(metrics, ignore_index=True)