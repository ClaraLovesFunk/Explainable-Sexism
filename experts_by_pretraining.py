class UCC_Dataset(Dataset): ########## WHY DO WE NEED FANCY DATASET MODULE?

  def __init__(self, data, tokenizer, attributes, max_token_len: int = 128, sample = None): ######## DEFINE HOW TO TRUNCATE SAMPLES,  THINK HOW MANY MAX TOKENS WE HAVE
    self.data = data
    self.tokenizer = tokenizer
    self.attributes = attributes
    self.max_token_len = max_token_len
    self.sample = sample
    self._prepare_data()

  def _prepare_data(self): 
    #data = pd.read_csv(self.data_path)
    #data['sexist'] = np.where(data['label_sexist'] == 'sexist', 1, 0)
    #labels = ['none','1. threats, plans to harm and incitement','2. derogation', '3. animosity','4. prejudiced discussions'] ##### TURN THIS into an input variable of the function!!!
    #for k in labels:
    #  data[k] = np.where(train_data['label_category'] == k, 1, 0)

    if self.sample is not None:                            
      sexist = data.loc[data['sexist']==1]
      clean = data.loc[data['sexist']==0]
      self.data = pd.concat([sexist.sample(self.sample, random_state=7), clean.sample(self.sample, random_state=7)])
    #else:
    #  self.data = data
    
  def __len__(self): 
    return len(self.data)

  def __getitem__(self, index):  
    item = self.data.iloc[index]
    comment = str(item.text)             
    attributes = torch.FloatTensor(item[self.attributes])
    tokens = self.tokenizer.encode_plus(comment,
                                        add_special_tokens=True,
                                        return_tensors='pt',
                                        truncation=True,
                                        padding='max_length',
                                        max_length=self.max_token_len,
                                        return_attention_mask = True)
    return {'input_ids': tokens.input_ids.flatten(), 'attention_mask': tokens.attention_mask.flatten(), 'labels': attributes}







class UCC_Data_Module(pl.LightningDataModule):

  def __init__(self,model_name, X_train, X_test, attributes, batch_size: int = 16, max_token_length: int = 128): #model_name='roberta-base' ######ADDEDCONFIG
    super().__init__()
    #self.config = config   ####### ADDED BY ME         
    self.X_train = X_train
    self.X_test = X_test
    self.attributes = attributes
    self.batch_size = batch_size
    self.max_token_length = max_token_length
    self.model_name = model_name
    self.tokenizer = AutoTokenizer.from_pretrained(model_name) #model_name

  def setup(self, stage = None):
    if stage in (None, "fit"): 
      self.train_dataset = UCC_Dataset(self.X_train, attributes=self.attributes, tokenizer=self.tokenizer, sample=None) ####### ADD SAMPLE PARAMETER
      self.val_dataset = UCC_Dataset(self.X_test, attributes=self.attributes, tokenizer=self.tokenizer, sample=None) ###### REMOVE SAMPLE PARAMETER
    if stage == 'test':
      self.test_dataset = UCC_Dataset(self.X_test, attributes=self.attributes, tokenizer=self.tokenizer, sample=None) 
    if stage == 'predict': ######### CAN WE DISTINGUISH BETWEEN PREDICT FOR VAL AND PREDICT FOR TEST????????????
      self.val_dataset = UCC_Dataset(self.X_test, attributes=self.attributes, tokenizer=self.tokenizer, sample=None) 

  def train_dataloader(self): ####### HERE ITS NICELY SEPERATED IN TRAIN, VAL, TEST -- WHY DIDNT WE DO THAT ABOVE IN SETUP????????
    return DataLoader(self.train_dataset, batch_size = self.batch_size, num_workers=4, shuffle=True) 

  def val_dataloader(self):
    return DataLoader(self.val_dataset, batch_size = self.batch_size, num_workers=4, shuffle=False)

  def predict_dataloader(self):
    return DataLoader(self.val_dataset, batch_size = self.batch_size, num_workers=4, shuffle=False)




class UCC_Comment_Classifier(pl.LightningModule):

  def __init__(self, config: dict):
    super().__init__()
    self.config = config
    self.pretrained_model = AutoModel.from_pretrained(config['model_name'], return_dict = True)
    self.hidden = torch.nn.Linear(self.pretrained_model.config.hidden_size, self.pretrained_model.config.hidden_size)
    self.classifier = torch.nn.Linear(self.pretrained_model.config.hidden_size, self.config['n_labels'])
    self.soft = torch.nn.Softmax(dim=1)
    torch.nn.init.xavier_uniform_(self.classifier.weight) # makes quicker
    self.loss_func = nn.BCEWithLogitsLoss(reduction='mean') ####### WE JUST WANT CROSSENTROPY??????????
    self.dropout = nn.Dropout()
    self.f1_func = MulticlassF1Score(num_classes = self.config['n_labels']) #########task='multiclass', average='macro'
    
  def forward(self, input_ids, attention_mask, labels=None):
    # roberta layer
    output = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
    pooled_output = torch.mean(output.last_hidden_state, 1) 
    # final logits
    pooled_output = self.dropout(pooled_output)
    pooled_output = self.hidden(pooled_output)
    pooled_output = F.relu(pooled_output)
    pooled_output = self.dropout(pooled_output)
    logits = self.classifier(pooled_output)
    logits = self.soft(logits)
    # calculate loss and f1
    loss = 0
    f1 = 0
    if labels is not None:
      loss = self.loss_func(logits.view(-1, self.config['n_labels']), labels.view(-1, self.config['n_labels'])) ##### WHY AND HOW DO WE NEED TO MAKE SURE THAT WHAT IS OF THE SAME SHAPE??? THE LOGITS?
      f1 = self.f1_func(logits.view(-1, self.config['n_labels']), labels.view(-1, self.config['n_labels']))     ########self.f1_func(logits, labels)  
    return loss, f1, logits

  def training_step(self, batch, batch_index):
    loss, f1, outputs = self(**batch)
    self.log("train f1 ", f1, prog_bar = True, logger=True)
    self.log("train loss ", loss, prog_bar = True, logger=True)
    return {"loss":loss, "train f1":f1, "predictions":outputs, "labels": batch["labels"]}
  
  def validation_step(self, batch, batch_index):
    loss, f1, outputs = self(**batch)
    self.log("val f1", f1, prog_bar = True, logger=True)
    self.log("val loss ", loss, prog_bar = True, logger=True)
    return {"val_loss": loss, "val f1":f1, "predictions":outputs, "labels": batch["labels"]}
  
  def predict_step(self, batch, batch_index):
    _, _, outputs = self(**batch)
    return outputs

  def configure_optimizers(self):
    optimizer = AdamW(self.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
    total_steps = self.config['train_size']/self.config['batch_size']
    warmup_steps = math.floor(total_steps * self.config['warmup'])
    warmup_steps = math.floor(total_steps * self.config['warmup'])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    return [optimizer],[scheduler]

  # def validation_epoch_end(self, outputs):
  #   losses = []
  #   for output in outputs:
  #     loss = output['val_loss'].detach().cpu()
  #     losses.append(loss)
  #   avg_loss = torch.mean(torch.stack(losses))
  #   self.log("avg_val_loss", avg_loss)



