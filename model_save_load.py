def model_export(self, sess, save_dir):
    """
    def train():
        ...
        with tf.Session() as sess:
            ...
            for e in range(args.num_epochs):
                for b in range(data_loader.num_batches):
                    ...
            print("[Train] train is over, total time is {}".format(time.time() - tic))
            model_export(sess, lstm_model)

    :param sess:
    :param save_dir:
    :return:
    """
    if os.path.exists(save_dir):
        cmd = "rm -rf %s" % save_dir
        logging.warning("[TextCNN][Export] %s" % cmd)
        os.system(cmd)

    logging.info("[TextCNN][Export] begin to export model for tf-servering")
    export_path = save_dir
    logging.info("[TextCNN][Export] export path is {}".format(export_path))
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    graph = tf.get_default_graph()

    # 明确模型的输入输出
    text = tf.saved_model.utils.build_tensor_info(
        tensor=graph.get_tensor_by_name("sentence:0"))
    # label_placeholder = tf.saved_model.utils.build_tensor_info(
    #     tensor=graph.get_tensor_by_name("label:0"))

    one_hot_prediction = tf.saved_model.utils.build_tensor_info(
        tensor=graph.get_tensor_by_name("output/one_hot_prediction:0"))
    # out_score = tf.saved_model.utils.build_tensor_info(
    #     tensor=graph.get_tensor_by_name("inference/output:0"))

    # 定义模型的签名
    prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs={
            'text': text,
            # 'label': label_placeholder
        },
        outputs={
            'one_hot_prediction': one_hot_prediction,
            # 'score': out_score,
        },
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

    # 需要和 tf-serving 后端对接
    # legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    builder.add_meta_graph_and_variables(
        sess,
        tags=[tf.saved_model.tag_constants.SERVING],
        signature_def_map={'prediction': prediction_signature},
        # legacy_init_op=legacy_init_op
    )
    builder.save()
    logging.info("[Export] Done exporting for tf-serving!!!")


def freeze_graph_test(pb_path, dialog):
    '''
    :param pb_path:pb文件的路径
    :param dialog: 通话文本
    :return:
    '''
    # with tf.Graph().as_default():
        # output_graph_def = tf.GraphDef()
        # with open(pb_path, "rb") as f:
        #     output_graph_def.ParseFromString(f.read())
        #     tf.import_graph_def(output_graph_def, name="")
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_dir=pb_path)
        sess.run(tf.global_variables_initializer())

        # 定义输入的张量名称,对应网络结构的输入张量
        text = sess.graph.get_tensor_by_name("sentence:0")

        # 定义输出的张量名称
        x_idx = sess.graph.get_tensor_by_name('input/x_idx:0')
        one_hot_prediction = sess.graph.get_tensor_by_name("output/one_hot_prediction:0")
        scores = sess.graph.get_tensor_by_name('output/scores:0')

        out = sess.run([x_idx, scores, one_hot_prediction], feed_dict={text: dialog})
        print("x_idx:{}, out:{}, scores:{}".format(x_idx, out, scores))

        
def load_model_from_checkpoint(dialog):
    model_dir = tfFLAGS.pretrain_dir
    if not tf.train.get_checkpoint_state(model_dir):
        raise ValueError("must supply pre-train model when you are testing !!!")

    session_conf = tf.ConfigProto(
        allow_soft_placement=tfFLAGS.allow_soft_placement,
        log_device_placement=tfFLAGS.log_device_placement
    )
    with tf.Session(config=session_conf) as sess:
        ckpt = tf.train.latest_checkpoint(checkpoint_dir=model_dir)
        graph_from_meta = ckpt + ".meta"
        saver = tf.train.import_meta_graph(graph_from_meta)
        saver.restore(sess, tf.train.latest_checkpoint(model_dir))

        graph = sess.graph
        feed_dict = {
            "sentence:0": dialog,
        }

        # # correct_num = graph.get_tensor_by_name('precision/correct_num:0')
        # # pred_num = graph.get_tensor_by_name('precision/pred_num:0')
        # one_hot_prediction = graph.get_tensor_by_name('output/one_hot_prediction:0')
        #
        # one_hot_prediction = sess.run(
        #     [one_hot_prediction], feed_dict=feed_dict)
        #
        # # print(correct_num)
        # # print(pred_num)
        # print(one_hot_prediction)

        x_idx = sess.graph.get_tensor_by_name('input/x_idx:0')
        one_hot_prediction = sess.graph.get_tensor_by_name("output/one_hot_prediction:0")
        scores = sess.graph.get_tensor_by_name('output/scores:0')

        x_idx_out, scores_out, one_hot_prediction_out = \
            sess.run([x_idx, scores, one_hot_prediction], feed_dict=feed_dict)
        print("x_idx_out:{}, scores_out:{}, one_hot_prediction_out:{}"
              .format(x_idx_out, scores_out, one_hot_prediction_out))
        

def ckpt_to_pb(ckpt_dir, pb_dir):
    logging.info("===============================")
    logging.info(tfFLAGS.flags_into_string())
    logging.info("===============================")
    dictionary = Dictionary(tfFLAGS.label_file,
                            tfFLAGS.vocab_file,
                            tfFLAGS.max_seq_len,
                            tfFLAGS.fc_hidden_size1,
                            tfFLAGS.fc_hidden_size2,
                            tfFLAGS.threshold)

    text_cnn_model = TextCNN(
        dictionary,
        embedding_size=tfFLAGS.embedding_dim,
        filter_sizes=list(map(int, tfFLAGS.filter_sizes.split(","))),
        num_filters=tfFLAGS.num_filters,
        l2_reg_lambda=tfFLAGS.l2_reg_lambda,
        dropout_keep_prob=tfFLAGS.dropout_keep_prob,
        learning_rate=tfFLAGS.learning_rate,
        warmup_steps=tfFLAGS.warmup_steps
    )

    if os.path.exists(pb_dir):
        cmd = "rm -rf %s" % pb_dir
        os.system(cmd)

    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        text_cnn_model.load(sess, ckpt_dir)

        with tf.device("/cpu:0"):
            insert_vocab_op = text_cnn_model.symbol2index.insert(
                keys=tf.constant(list(text_cnn_model.vocab.keys()), dtype=tf.string),
                values=tf.constant(list(text_cnn_model.vocab.values()), dtype=tf.int64))
            sess.run([insert_vocab_op])

        logging.info('Saving pb model to %s!' % pb_dir)
        text_cnn_model.model_export(sess, pb_dir)
        
        
if __name__ == '__main__':
    pb_path = 'model\\export'

    # config
    PAD = '<pad>'
    EOS = '<eos>'
    UNK = '<unk>'
    UNK_LABEL = 'OTHER'
    delimiter = '\t'
    label_delimiter = '-'

    # 输入ckpt模型路径
    input_checkpoint = temp + 'model.ckpt-310'
    # 输出pb模型的路径
    out_pb_path = out + "/frozen_model.pb"

    if 1:
        # 调用freeze_graph将ckpt转为pb
        freeze_graph(input_checkpoint, out_pb_path)

    # pb 模型测试
    dialog = []
    # for line in open(out + '/dialog.txt', 'r', encoding='utf-8'):
    #     line = padding(line.strip(), 600, EOS, PAD)
    for line in open(out + '/valid_data.txt', 'r', encoding='utf-8'):
        line = padding(line.strip().split('\t')[1], 600, EOS, PAD)
        print(line)
        dialog.append(line)
    freeze_graph_test(pb_path, dialog)
