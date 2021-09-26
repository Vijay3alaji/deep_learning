

import numpy as np
import tensorflow.compat.v1 as tf
import re
import time

lines = open("movie_lines.txt",encoding='utf-8',errors='ignore').read().split("\n")
conversation = open("movie_conversations.txt",encoding='utf-8',errors='ignore').read().split("\n")

lineid = {}
for line in lines:
    _temp = line.split(" +++$+++ ")
    if(len(_temp)==5):
        lineid[_temp[0]]=_temp[4]  
        
convoid = []
for conv in conversation[:-1]:
    _temp1 = conv.split(" +++$+++ ")[-1][1:-1].replace("'","").replace(" ","")
    convoid.append(_temp1.split(","))
    

questions = []
answers = []

for conv in convoid:
    for i in range(len(conv) - 1):
        questions.append(lineid[conv[i]])
        answers.append(lineid[conv[i+1]])
        
        
def cleansing(text):
    text = text.lower()
    text = re.sub(r"i'm","i am",text)
    text = re.sub(r"he's","he is",text)
    text = re.sub(r"she's","she is",text)
    text = re.sub(r"that's","that is",text)
    text = re.sub(r"what's","what is",text)
    text = re.sub(r"where's","where is",text)
    text = re.sub(r"\'ll"," will",text)
    text = re.sub(r"\'ve"," have",text)
    text = re.sub(r"\'re"," are",text)
    text = re.sub(r"\'d"," would",text)
    text = re.sub(r"can't","cannot",text)
    text = re.sub(r"won't","will not",text)
    text = re.sub(r"it's","it is",text)
    text = re.sub(r"\'bout"," about",text)
    text = re.sub(r"workin","working",text)
    text = re.sub(r"[-()\"#/@:;<>{}+=|&.?,]","",text)
    return text
    
    
cleansed_question = []
for question in questions:
    cleansed_question.append(cleansing(question))
    
    
cleansed_answer = []
for answer in answers:
    cleansed_answer.append(cleansing(answer))


short_questions = []
short_answers = []
i = 0
for question in cleansed_question:
    if 2 <= len(question.split()) <= 25:
        short_questions.append(question)
        short_answers.append(cleansed_answer[i])
    i += 1
    
cleansed_question = []
cleansed_answer = []
i = 0
for answer in short_answers:
    if 2 <= len(answer.split()) <= 25:
        cleansed_answer.append(answer)
        cleansed_question.append(short_questions[i])
    i += 1

    
wordcount = {}
for question in cleansed_question:
    for word in question.split():
        if word not in wordcount:
            wordcount[word]=1
        else:
            wordcount[word]+=1
            
for answer in cleansed_answer:
    for word in answer.split():
        if word not in wordcount:
            wordcount[word]=1
        else:
            wordcount[word]+=1
            
threshold = 15
questionword2int={}
word_number=0
for word,count in wordcount.items():
    if count >= threshold:
        questionword2int[word]=word_number
        word_number+=1
        
answerword2int={}
word_number=0
for word,count in wordcount.items():
    if count >= threshold:
        answerword2int[word]=word_number
        word_number+=1


tokens = ["<PAD>","<EOS>","<OUT>","<SOS>"]
for token in tokens:
    questionword2int[token]=len(questionword2int)+1
for token in tokens:
    answerword2int[token]=len(answerword2int)+1
    
answerint2word = {w_i:w for w , w_i in answerword2int.items()}    

    
for i in range(len(cleansed_answer)):
    cleansed_answer[i] += " <EOS>"
    
questions_to_int=[]
for question in cleansed_question:
    ints=[]
    for word in question.split():
        if word not in questionword2int:
            ints.append(questionword2int["<OUT>"])
        else:
            ints.append(questionword2int[word])
    questions_to_int.append(ints)
    
answers_to_int=[]
for answer in cleansed_answer:
    ints=[]
    for word in answer.split():
        if word not in answerword2int:
            ints.append(answerword2int["<OUT>"])
        else:
            ints.append(answerword2int[word])
    answers_to_int.append(ints)   
    
    
sorted_cleansed_question = []
sorted_cleansed_answer = []
for length in range(1,25 +1):
    for i in enumerate(questions_to_int):
        if len(i[1]) == length:
            sorted_cleansed_question.append(questions_to_int [i[0]])
            sorted_cleansed_answer.append(answers_to_int [i[0]])
            
        
def model_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name = 'input')
    targets = tf.placeholder(tf.int32, [None, None], name = 'target')
    lr = tf.placeholder(tf.float32, name = 'learning_rate')
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
    return inputs, targets, lr, keep_prob   
    
    
def preprocess_targets(targets, word2int, batch_size):
    left_side = tf.fill([batch_size, 1], word2int['<SOS>'])
    right_side = tf.strided_slice(targets, [0,0], [batch_size, -1], [1,1])
    preprocessed_targets = tf.concat([left_side, right_side], 1)
    return preprocessed_targets    
    
def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell,
                    cell_bw = encoder_cell, sequence_length = sequence_length,
                    inputs = rnn_inputs,dtype = tf.float32)
    return encoder_state   
    
    
def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = "bahdanau", num_units = decoder_cell.output_size)
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
    attention_keys,attention_values,attention_score_function,attention_construct_function,name = "attn_dec_train")
    decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
    training_decoder_function,decoder_embedded_input,sequence_length,scope = decoding_scope)
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    
    return output_function(decoder_output_dropout)
    
    
    
def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, maximum_length, num_words, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = "bahdanau", num_units = decoder_cell.output_size)
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                              encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              decoder_embeddings_matrix,
                                                                              sos_id,
                                                                              eos_id,
                                                                              maximum_length,
                                                                              num_words,
                                                                              name = "attn_dec_inf")
    test_predictions, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,test_decoder_function,scope = decoding_scope)
                                                                                                                
                                                                                                                
    return test_predictions
    

def decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size):
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
        weights = tf.truncated_normal_initializer(stddev = 0.1)
        biases = tf.zeros_initializer()
        output_function = lambda x: tf.contrib.layers.fully_connected(x,
                                                                      num_words,
                                                                      None,
                                                                      scope = decoding_scope,
                                                                      weights_initializer = weights,
                                                                      biases_initializer = biases)
        training_predictions = decode_training_set(encoder_state,
                                                   decoder_cell,
                                                   decoder_embedded_input,
                                                   sequence_length,
                                                   decoding_scope,
                                                   output_function,
                                                   keep_prob,
                                                   batch_size)
        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(encoder_state,
                                           decoder_cell,
                                           decoder_embeddings_matrix,
                                           word2int['<SOS>'],
                                           word2int['<EOS>'],
                                           sequence_length - 1,
                                           num_words,
                                           decoding_scope,
                                           output_function,
                                           keep_prob,
                                           batch_size)
    return training_predictions, test_predictions

def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words, questions_num_words, encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, questionword2int):
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs,
                                                              answers_num_words + 1,
                                                              encoder_embedding_size,
                                                              initializer = tf.random_uniform_initializer(0, 1))
    encoder_state = encoder_rnn(encoder_embedded_input, rnn_size, num_layers, keep_prob, sequence_length)
    preprocessed_targets = preprocess_targets(targets, questionword2int, batch_size)
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words + 1, decoder_embedding_size], 0, 1))
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_input,
                                                         decoder_embeddings_matrix,
                                                         encoder_state,
                                                         questions_num_words,
                                                         sequence_length,
                                                         rnn_size,
                                                         num_layers,
                                                         questionword2int,
                                                         keep_prob,
                                                         batch_size)
    return training_predictions, test_predictions


epochs = 100
batch_size = 32
rnn_size = 1024
num_layers = 3
encoding_embedding_size = 1024
decoding_embedding_size = 1024
learning_rate = 0.001
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.5

tf.reset_default_graph()
session = tf.InteractiveSession()

inputs, targets, lr, keep_prob = model_inputs()

sequence_length = tf.placeholder_with_default(25, None, name = 'sequence_length')

input_shape = tf.shape(inputs)

training_predictions, test_predictions = seq2seq_model(tf.reverse(inputs, [-1]),
                                                       targets,
                                                       keep_prob,
                                                       batch_size,
                                                       sequence_length,
                                                       len(answerword2int),
                                                       len(questionword2int),
                                                       encoding_embedding_size,
                                                       decoding_embedding_size,
                                                       rnn_size,
                                                       num_layers,
                                                       questionword2int)

with tf.name_scope("optimization"):
    loss_error = tf.contrib.seq2seq.sequence_loss(training_predictions,
                                                  targets,
                                                  tf.ones([input_shape[0], sequence_length]))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = optimizer.compute_gradients(loss_error)
    clipped_gradients = [(tf.clip_by_value(grad_tensor, -5., 5.), grad_variable) for grad_tensor, grad_variable in gradients if grad_tensor is not None]
    optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradients)
    
def apply_padding(batch_of_sequences, word2int):
    max_sequence_length = max([len(sequence) for sequence in batch_of_sequences])
    return [sequence + [word2int['<PAD>']] * (max_sequence_length - len(sequence)) for sequence in batch_of_sequences]


def split_into_batches(questions, answers, batch_size):
    for batch_index in range(0, len(questions) // batch_size):
        start_index = batch_index * batch_size
        questions_in_batch = questions[start_index : start_index + batch_size]
        answers_in_batch = answers[start_index : start_index + batch_size]
        padded_questions_in_batch = np.array(apply_padding(questions_in_batch, questionword2int))
        padded_answers_in_batch = np.array(apply_padding(answers_in_batch, answerword2int))
        yield padded_questions_in_batch, padded_answers_in_batch

training_validation_split = int(len(sorted_cleansed_question) * 0.15)
training_questions = sorted_cleansed_question[training_validation_split:]
training_answers = sorted_cleansed_answer[training_validation_split:]
validation_questions = sorted_cleansed_question[:training_validation_split]
validation_answers = sorted_cleansed_answer[:training_validation_split]

batch_index_check_training_loss = 100
batch_index_check_validation_loss = ((len(training_questions)) // batch_size // 2) - 1
total_training_loss_error = 0
list_validation_loss_error = []
early_stopping_check = 0
early_stopping_stop = 100
checkpoint = "chatbot_weights.ckpt"
session.run(tf.global_variables_initializer())
for epoch in range(1, epochs + 1):
    for batch_index, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(training_questions, training_answers, batch_size)):
        starting_time = time.time()
        _, batch_training_loss_error = session.run([optimizer_gradient_clipping, loss_error], {inputs: padded_questions_in_batch,
                                                                                               targets: padded_answers_in_batch,
                                                                                               lr: learning_rate,
                                                                                               sequence_length: padded_answers_in_batch.shape[1],
                                                                                               keep_prob: keep_probability})
        total_training_loss_error += batch_training_loss_error
        ending_time = time.time()
        batch_time = ending_time - starting_time
        if batch_index % batch_index_check_training_loss == 0:
            print('Epoch: {:>3}/{}, Batch: {:>4}/{}, Training Loss Error: {:>6.3f}, Training Time on 100 Batches: {:d} seconds'.format(epoch,
                                                                                                                                       epochs,
                                                                                                                                       batch_index,
                                                                                                                                       len(training_questions) // batch_size,
                                                                                                                                       total_training_loss_error / batch_index_check_training_loss,
                                                                                                                                       int(batch_time * batch_index_check_training_loss)))
            total_training_loss_error = 0
        if batch_index % batch_index_check_validation_loss == 0 and batch_index > 0:
            total_validation_loss_error = 0
            starting_time = time.time()
            for batch_index_validation, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(validation_questions, validation_answers, batch_size)):
                batch_validation_loss_error = session.run(loss_error, {inputs: padded_questions_in_batch,
                                                                       targets: padded_answers_in_batch,
                                                                       lr: learning_rate,
                                                                       sequence_length: padded_answers_in_batch.shape[1],
                                                                       keep_prob: 1})
                total_validation_loss_error += batch_validation_loss_error
            ending_time = time.time()
            batch_time = ending_time - starting_time
            average_validation_loss_error = total_validation_loss_error / (len(validation_questions) / batch_size)
            print('Validation Loss Error: {:>6.3f}, Batch Validation Time: {:d} seconds'.format(average_validation_loss_error, int(batch_time)))
            learning_rate *= learning_rate_decay
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate
            list_validation_loss_error.append(average_validation_loss_error)
            if average_validation_loss_error <= min(list_validation_loss_error):
                print('I speak better now!!')
                early_stopping_check = 0
                saver = tf.train.Saver()
                saver.save(session, checkpoint)
            else:
                print("Sorry I do not speak better, I need to practice more.")
                early_stopping_check += 1
                if early_stopping_check == early_stopping_stop:
                    break
    if early_stopping_check == early_stopping_stop:
        print("My apologies, I cannot speak better anymore. This is the best I can do.")
        break
print("Game Over")


checkpoint = "./chatbot_weights.ckpt"
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(session, checkpoint)

def convert_string2int(question, word2int):
    question = cleansing(question)
    return [word2int.get(word, word2int['<OUT>']) for word in question.split()]

while(True):
    question = input("You: ")
    if question == 'Goodbye':
        break
    question = convert_string2int(question, questionword2int)
    question = question + [questionword2int['<PAD>']] * (25 - len(question))
    fake_batch = np.zeros((batch_size, 25))
    fake_batch[0] = question
    predicted_answer = session.run(test_predictions, {inputs: fake_batch, keep_prob: 0.5})[0]
    answer = ''
    for i in np.argmax(predicted_answer, 1):
        if answerint2word[i] == 'i':
            token = ' I'
        elif answerint2word[i] == '<EOS>':
            token = '.'
        elif answerint2word[i] == '<OUT>':
            token = 'out'
        else:
            token = ' ' + answerint2word[i]
        answer += token
        if token == '.':
            break
    print('ChatBot: ' + answer)















 
        
