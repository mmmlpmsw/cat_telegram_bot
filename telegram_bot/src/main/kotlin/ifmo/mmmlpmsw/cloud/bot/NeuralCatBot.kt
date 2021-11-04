package ifmo.mmmlpmsw.cloud.bot

import org.telegram.telegrambots.bots.TelegramLongPollingBot
import org.telegram.telegrambots.meta.api.methods.AnswerInlineQuery
import org.telegram.telegrambots.meta.api.methods.send.SendMessage
import org.telegram.telegrambots.meta.api.objects.Update
import org.telegram.telegrambots.meta.api.objects.inlinequery.InlineQuery
import org.telegram.telegrambots.meta.api.objects.inlinequery.result.InlineQueryResult
import org.telegram.telegrambots.meta.api.objects.inlinequery.result.InlineQueryResultPhoto
import org.telegram.telegrambots.meta.exceptions.TelegramApiException
import java.util.*
import kotlin.collections.ArrayList


class NeuralCatBot: TelegramLongPollingBot() {

    private val CACHETIME = 0
    override fun getBotToken(): String = CatBotProperties.botToken
    override fun getBotUsername(): String = CatBotProperties.botUsername
    override fun onUpdateReceived(update: Update?) {
//        if (update?.message != null && update.message.text != null) {
//            val message: String? = update.message.text
//            sendMsg(update.message.chatId.toString(), message)
//        }
        if (update!!.hasInlineQuery()) {
            processQuery(update.inlineQuery)
        }
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

    @Synchronized
    fun processQuery(inlineQuery: InlineQuery) {
        val query = inlineQuery.query
        try {
            if (query.isNotEmpty()) {
                execute(converteResultsToResponse(inlineQuery))
            }
        } catch (e: TelegramApiException) {
            e.printStackTrace()
        }
    }

    private fun converteResultsToResponse(inlineQuery: InlineQuery): AnswerInlineQuery {
        val answerInlineQuery = AnswerInlineQuery()
        answerInlineQuery.inlineQueryId = inlineQuery.id
        answerInlineQuery.cacheTime = CACHETIME
        answerInlineQuery.results = convertResults()
        return answerInlineQuery
    }

    private fun convertResults(): List<InlineQueryResult> {
        val inlineQueryResults: MutableList<InlineQueryResult> = ArrayList()
        inlineQueryResults.add(InlineQueryResultPhoto(
            UUID.randomUUID().toString(),
            "https://cs12.pikabu.ru/post_img/2020/12/07/1/1607295606111935536.jpg"
        ).also {
            it.thumbUrl = "https://cs12.pikabu.ru/post_img/2020/12/07/1/1607295606111935536.jpg"
        })
        return inlineQueryResults
    }

}