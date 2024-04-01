model_checkpoint = ModelCheckpoint(filepath='best_model.h5',  # Fixed filepath extension
                                   monitor='val_loss',
                                   save_best_only=True)