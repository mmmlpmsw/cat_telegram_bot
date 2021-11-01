package ifmo.mmmlpmsw.cloud

import org.telegram.telegrambots.bots.TelegramLongPollingBot
import org.telegram.telegrambots.meta.api.methods.send.SendMessage
import org.telegram.telegrambots.meta.api.objects.Update
import org.telegram.telegrambots.meta.exceptions.TelegramApiException


class NeuralCatBot: TelegramLongPollingBot() {

    override fun getBotToken(): String = CatBotProperties.botToken
    override fun getBotUsername(): String = CatBotProperties.botUsername
    override fun onUpdateReceived(update: Update?) {
        val message: String? = update?.message?.text
        sendMsg(update?.message?.chatId.toString(), message)
    }

    @Synchronized
    fun sendMsg(chatId: String?, s: String?) {
        val message = SendMessage.builder().chatId(chatId!!).text("$s aaaaaaaaaa").build()
        message.enableMarkdown(true)
        try {
            execute(message)
        } catch (e: TelegramApiException) {
            e.printStackTrace()
        }

    }

}